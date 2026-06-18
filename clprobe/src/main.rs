//! clprobe: LLInt-templates-via-Cranelift codegen probe.
//!
//! For each of 5 representative AOT bodies, build CLIF via FunctionBuilder
//! using per-bytecode-opcode emitters that mirror LLInt handler semantics
//! (NOT the C-AOT helper logic). Slow-path tails are extern `preserve_all`
//! calls. The point of the probe is to inspect regalloc2's choices: do
//! cfr / globalObject / mdBase / numberTag / notCellMask survive bridge calls
//! in callee-saved regs without per-site spill?
//!
//! Output (per body, into $OUTDIR):
//!   <name>.clif      pretty-printed CLIF (post-build, pre-opt)
//!   <name>.vcode     post-regalloc disasm (set_disasm=true)
//!   <name>.o         ELF object via ObjectModule
//!   summary.txt      ops / CLIF-instrs / mach-instrs / .text-bytes table

#![allow(dead_code, unused_variables, clippy::too_many_arguments)]

use anyhow::Result;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{I32, I64, I8};
use cranelift_codegen::ir::{
    AbiParam, Block, FuncRef, InstBuilder, MemFlagsData, UserFuncName, Value,
};
use cranelift_codegen::isa::{CallConv, TargetFrontendConfig};

#[inline(always)] fn mt() -> MemFlagsData { MemFlagsData::trusted() }
#[inline(always)] fn ro() -> MemFlagsData { MemFlagsData::trusted().with_readonly() }
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// JSC ABI / layout constants (x86_64, JSVALUE64). Offsets verified against
// LLIntAssembly.h / generated offsets in build dir; for the probe the EXACT
// offsets do not affect register-allocation shape, only displacement encoding.
// ---------------------------------------------------------------------------

const NUMBER_TAG: i64 = 0xfffe_0000_0000_0000u64 as i64;
const NOT_CELL_MASK: i64 = NUMBER_TAG | 2; // 0xfffe_0000_0000_0002
const JS_UNDEFINED: i64 = 0x0a;
const JS_NULL: i64 = 0x02;
const JS_TRUE: i64 = 0x07;
const JS_FALSE: i64 = 0x06;
const BOOL_TAG: i64 = 0x04;

// CallFrame slot offsets (8-byte slots, indices from CallFrameSlot)
const CFR_CODEBLOCK: i32 = 2 * 8;
const CFR_CALLEE: i32 = 3 * 8;
const CFR_ARGCOUNT: i32 = 4 * 8;
const CFR_THIS: i32 = 5 * 8;
const CFR_ARG0: i32 = 6 * 8;
const CFR_CALLSITE: i32 = 4 * 8 + 4; // tag word of argumentCountIncludingThis slot

// Derived-pointer offsets (representative; structureID at JSCell+0, type at +5,
// butterfly at +8, JSCallee::scope at +16, etc.)
const JSCELL_SID: i32 = 0;
const JSCELL_TYPE: i32 = 5;
const JSCELL_INDEXING: i32 = 6;
const JSOBJ_BUTTERFLY: i32 = 8;
const JSCALLEE_SCOPE: i32 = 16;
const GO_VM_OFFSET: i32 = 0x38; // JSGlobalObject::m_vm
const VM_EXC_OFFSET: i32 = 0x1f508; // VM::m_exception (representative)
const VM_TRAPS_OFFSET: i32 = 0x28a40; // VM::m_traps.m_trapBits
const CB_METADATA: i32 = 0x88; // CodeBlock::m_metadata

// Slow-path symbol ids (declare_func_in_module index space). All preserve_all.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
enum Slow {
    GetById,        // (go, cfr, base, bcOff, identIdx) -> i64
    PutById,        // (go, cfr, base, val, bcOff, identIdx) -> void(i64)
    GetByVal,       // (go, cfr, base, idx, bcOff) -> i64
    PutByVal,       // (go, cfr, base, idx, val, bcOff) -> i64
    GetLength,      // (go, cfr, base, bcOff) -> i64
    Add,            // (go, lhs, rhs) -> i64
    Less,           // (go, lhs, rhs) -> i64 (bool)
    Inc,            // (go, v) -> i64
    BitOp,          // (go, lhs, rhs) -> i64
    ToThis,         // (go, cfr, this) -> i64
    ResolveScope,   // (go, cfr, mdOff) -> i64
    GetFromScope,   // (go, cfr, scope, mdOff) -> i64
    PutToScope,     // (go, cfr, scope, val, mdOff) -> i64
    DispatchCall,   // (go, cfr, callee, this, calleeFrame, argc, bcOff) -> i64  ** SystemV, NOT preserve_all
    HandleTraps,    // (go) -> i64
    StrictEq,       // (go, lhs, rhs) -> i64
    ToBoolean,      // (go, v) -> i64
}

// ---------------------------------------------------------------------------
// Bytecode model (just enough for the 5 bodies)
// ---------------------------------------------------------------------------

/// Virtual register: positive = arg index (1-based), negative = local index, 0 = `this`.
type VR = i32;

#[derive(Clone, Debug)]
enum Op {
    Enter,
    Mov(VR, VR),                        // dst, src
    MovK(VR, i64),                      // dst, encoded JSValue const
    ToThis,
    GetById { dst: VR, base: VR, md_off: i32, ident: u32, bc: u32 },
    PutById { base: VR, val: VR, md_off: i32, ident: u32, bc: u32 },
    GetByVal { dst: VR, base: VR, idx: VR, bc: u32 },
    PutByVal { base: VR, idx: VR, val: VR, bc: u32 },
    GetLength { dst: VR, base: VR, bc: u32 },
    Add { dst: VR, lhs: VR, rhs: VR, bc: u32 },
    AddK { dst: VR, lhs: VR, k: i32, bc: u32 },
    Inc { dst: VR, bc: u32 },
    BitOp { kind: u8, dst: VR, lhs: VR, rhs: VR, bc: u32 },     // 0=and 1=or 2=xor 3=rshift
    BitOpK { kind: u8, dst: VR, lhs: VR, k: i32, bc: u32 },
    ResolveScope { dst: VR, scope: VR, md_off: i32, bc: u32 },
    GetFromScope { dst: VR, scope: VR, md_off: i32, bc: u32 },
    PutToScope { scope: VR, val: VR, md_off: i32, bc: u32 },
    Call { dst: VR, callee: VR, this: VR, args: Vec<VR>, frame_size: i32, bc: u32 },
    CheckTraps { bc: u32 },
    LoopHint,
    JTrue { cond: VR, target: u32 },
    JFalse { cond: VR, target: u32 },
    JEqNull { v: VR, target: u32 },
    JNEqNull { v: VR, target: u32 },
    JStrictEq { lhs: VR, rhs: VR, target: u32 },
    JNStrictEq { lhs: VR, rhs: VR, target: u32 },
    JStrictEqK { lhs: VR, k: i64, target: u32 },
    JNStrictEqK { lhs: VR, k: i64, target: u32 },
    JLess { lhs: VR, rhs: VR, target: u32, bc: u32 },
    JNLess { lhs: VR, rhs: VR, target: u32, bc: u32 },
    NeqNull { dst: VR, src: VR },
    Jmp(u32),
    Label(u32),
    Ret(Option<VR>),
    RetK(i64),
}

struct Body {
    name: &'static str,
    n_args: u32,
    n_locals: u32,
    ops: Vec<Op>,
}

// ---------------------------------------------------------------------------
// Emitter context
// ---------------------------------------------------------------------------

struct Emit<'a, 'b> {
    b: &'b mut FunctionBuilder<'a>,
    // pinned Variables (so regalloc can choose; SSA builder hoists once)
    go: Variable,
    cfr: Variable,
    md: Variable,
    vm: Variable,
    ntag: Variable,
    ncm: Variable,
    // virtual registers
    vthis: Variable,
    args: Vec<Variable>,
    locs: Vec<Variable>,
    // labels (bytecode-offset -> block)
    labels: HashMap<u32, Block>,
    // common exception-return sink
    exc_sink: Block,
    // extern slow-path FuncRefs
    slows: HashMap<Slow, FuncRef>,
    // shared stack slot for preserve_all out-param (one slot reused — slow
    // paths don't nest)
    out_slot: cranelift_codegen::ir::StackSlot,
    // are slow-paths preserve_all (out-pointer ABI) or systemv (return value)?
    pa: bool,
}

impl<'a, 'b> Emit<'a, 'b> {
    fn vr(&self, r: VR) -> Variable {
        if r == 0 { self.vthis }
        else if r > 0 { self.args[(r - 1) as usize] }
        else { self.locs[(-r - 1) as usize] }
    }
    fn use_(&mut self, r: VR) -> Value { let v = self.vr(r); self.b.use_var(v) }
    fn def_(&mut self, r: VR, v: Value) { let var = self.vr(r); self.b.def_var(var, v); }

    fn go(&mut self) -> Value { self.b.use_var(self.go) }
    fn cfr(&mut self) -> Value { self.b.use_var(self.cfr) }
    fn md(&mut self) -> Value { self.b.use_var(self.md) }
    fn vm(&mut self) -> Value { self.b.use_var(self.vm) }
    fn ntag(&mut self) -> Value { self.b.use_var(self.ntag) }
    fn ncm(&mut self) -> Value { self.b.use_var(self.ncm) }

    fn label(&mut self, id: u32) -> Block {
        *self.labels.entry(id).or_insert_with(|| self.b.create_block())
    }

    /// Call a slow path that conceptually returns i64. In preserve_all mode,
    /// the return is via an appended out-pointer (stack slot). In systemv
    /// mode it's the call's direct result.
    fn slow_call_ret(&mut self, s: Slow, args: &[Value]) -> Value {
        let fr = self.slows[&s];
        if self.pa {
            let outp = self.b.ins().stack_addr(I64, self.out_slot, 0);
            let mut a: Vec<Value> = args.to_vec();
            a.push(outp);
            self.b.ins().call(fr, &a);
            self.b.ins().stack_load(I64, I64, self.out_slot, 0)
        } else {
            let call = self.b.ins().call(fr, args);
            self.b.inst_results(call)[0]
        }
    }
    fn slow_call_void(&mut self, s: Slow, args: &[Value]) {
        let fr = self.slows[&s];
        self.b.ins().call(fr, args);
    }

    fn stamp_callsite(&mut self, bc: u32) {
        let cfr = self.cfr();
        let k = self.b.ins().iconst(I32, bc as i64);
        self.b.ins().store(mt(), k, cfr, CFR_CALLSITE);
    }

    fn check_exc_and_continue(&mut self, cont: Block) {
        // if (vm->m_exception) goto exc_sink else goto cont
        let vm = self.vm();
        let exc = self.b.ins().load(I64, mt(), vm, VM_EXC_OFFSET);
        let exc_sink = self.exc_sink;
        self.b.ins().brif(exc, exc_sink, &[], cont, &[]);
    }

    /// Encode int32 JSValue: (i64)i | NUMBER_TAG
    fn box_int32(&mut self, lo: Value) -> Value {
        let w = self.b.ins().sextend(I64, lo);
        let tag = self.ntag();
        self.b.ins().bor(w, tag)
    }
}

// ---------------------------------------------------------------------------
// Per-opcode emitters. Each maps the LLInt handler shape: inline fast path,
// branch to a COLD block that stamps callsite + calls preserve_all slow path
// + checks exception, then merge. All blocks created here; caller passes the
// fallthrough continuation.
// ---------------------------------------------------------------------------

fn emit_get_by_id(e: &mut Emit, dst: VR, base: VR, md_off: i32, ident: u32, bc: u32, cont: Block) {
    // LLInt mono-IC: load metadata.structureID, compare base->structureID.
    let bv = e.use_(base);
    let ncm = e.ncm();
    let notcell = e.b.ins().band(bv, ncm);
    let cell_blk = e.b.create_block();
    let slow_blk = e.b.create_block();
    let hit_blk = e.b.create_block();
    let merge = e.b.create_block();
    e.b.append_block_param(merge, I64);
    e.b.set_cold_block(slow_blk);
    e.b.ins().brif(notcell, slow_blk, &[], cell_blk, &[]);

    // cell: compare structureID
    e.b.switch_to_block(cell_blk);
    let md = e.md();
    let cached_sid = e.b.ins().load(I32, ro(), md, md_off + 0);
    let base_sid = e.b.ins().load(I32, mt(), bv, JSCELL_SID);
    let eq = e.b.ins().icmp(IntCC::Equal, base_sid, cached_sid);
    e.b.ins().brif(eq, hit_blk, &[], slow_blk, &[]);

    // hit: load via cached offset (data-IC)
    e.b.switch_to_block(hit_blk);
    let off = e.b.ins().load(I32, ro(), md, md_off + 4);
    let off64 = e.b.ins().sextend(I64, off);
    let scaled = e.b.ins().ishl_imm(off64, 3);
    let addr = e.b.ins().iadd(bv, scaled);
    let val = e.b.ins().load(I64, mt(), addr, 16); // inlineStorage()[off]
    e.b.ins().jump(merge, &[val.into()]);

    // slow
    e.b.switch_to_block(slow_blk);
    e.stamp_callsite(bc);
    let go = e.go(); let cfr = e.cfr();
    let bc_v = e.b.ins().iconst(I32, bc as i64);
    let id_v = e.b.ins().iconst(I32, ident as i64);
    let r = e.slow_call_ret(Slow::GetById, &[go, cfr, bv, bc_v, id_v]);
    let merge2 = e.b.create_block();
    e.check_exc_and_continue(merge2);
    e.b.switch_to_block(merge2);
    e.b.ins().jump(merge, &[r.into()]);

    // merge → write dst → continue
    e.b.switch_to_block(merge);
    let out = e.b.block_params(merge)[0];
    e.def_(dst, out);
    e.b.ins().jump(cont, &[]);
}

fn emit_put_by_id(e: &mut Emit, base: VR, val: VR, md_off: i32, ident: u32, bc: u32, cont: Block) {
    let bv = e.use_(base);
    let vv = e.use_(val);
    let ncm = e.ncm();
    let notcell = e.b.ins().band(bv, ncm);
    let cell_blk = e.b.create_block();
    let slow_blk = e.b.create_block();
    let hit_blk = e.b.create_block();
    e.b.set_cold_block(slow_blk);
    e.b.ins().brif(notcell, slow_blk, &[], cell_blk, &[]);

    e.b.switch_to_block(cell_blk);
    let md = e.md();
    let cached_sid = e.b.ins().load(I32, ro(), md, md_off + 0);
    let base_sid = e.b.ins().load(I32, mt(), bv, JSCELL_SID);
    let eq = e.b.ins().icmp(IntCC::Equal, base_sid, cached_sid);
    e.b.ins().brif(eq, hit_blk, &[], slow_blk, &[]);

    e.b.switch_to_block(hit_blk);
    let off = e.b.ins().load(I32, ro(), md, md_off + 4);
    let off64 = e.b.ins().sextend(I64, off);
    let scaled = e.b.ins().ishl_imm(off64, 3);
    let addr = e.b.ins().iadd(bv, scaled);
    e.b.ins().store(mt(), vv, addr, 16);
    e.b.ins().jump(cont, &[]);

    e.b.switch_to_block(slow_blk);
    e.stamp_callsite(bc);
    let go = e.go(); let cfr = e.cfr();
    let bc_v = e.b.ins().iconst(I32, bc as i64);
    let id_v = e.b.ins().iconst(I32, ident as i64);
    e.slow_call_void(Slow::PutById, &[go, cfr, bv, vv, bc_v, id_v]);
    e.check_exc_and_continue(cont);
}

fn emit_get_by_val(e: &mut Emit, dst: VR, base: VR, idx: VR, bc: u32, cont: Block) {
    // LLInt fast: base isCell, idx isInt32, indexingType is Contiguous, in-bounds.
    let bv = e.use_(base);
    let iv = e.use_(idx);
    let ncm = e.ncm();
    let ntag = e.ntag();
    let notcell = e.b.ins().band(bv, ncm);
    let s1 = e.b.create_block(); let s2 = e.b.create_block(); let s3 = e.b.create_block();
    let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let merge = e.b.create_block(); e.b.append_block_param(merge, I64);
    e.b.ins().brif(notcell, slow, &[], s1, &[]);

    e.b.switch_to_block(s1);
    // isInt32: (idx & numberTag) == numberTag
    let tagbits = e.b.ins().band(iv, ntag);
    let isint = e.b.ins().icmp(IntCC::Equal, tagbits, ntag);
    e.b.ins().brif(isint, s2, &[], slow, &[]);

    e.b.switch_to_block(s2);
    let idx32 = e.b.ins().ireduce(I32, iv);
    let bf = e.b.ins().load(I64, mt(), bv, JSOBJ_BUTTERFLY);
    let publen = e.b.ins().load(I32, mt(), bf, -8); // publicLength
    let inb = e.b.ins().icmp(IntCC::UnsignedLessThan, idx32, publen);
    e.b.ins().brif(inb, s3, &[], slow, &[]);

    e.b.switch_to_block(s3);
    let idx64 = e.b.ins().uextend(I64, idx32);
    let scaled = e.b.ins().ishl_imm(idx64, 3);
    let addr = e.b.ins().iadd(bf, scaled);
    let val = e.b.ins().load(I64, mt(), addr, 0);
    // hole check: val != 0
    e.b.ins().brif(val, merge, &[val.into()], slow, &[]);

    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go(); let cfr = e.cfr();
    let bc_v = e.b.ins().iconst(I32, bc as i64);
    let r = e.slow_call_ret(Slow::GetByVal, &[go, cfr, bv, iv, bc_v]);
    let post = e.b.create_block();
    e.check_exc_and_continue(post);
    e.b.switch_to_block(post);
    e.b.ins().jump(merge, &[r.into()]);

    e.b.switch_to_block(merge);
    let out = e.b.block_params(merge)[0];
    e.def_(dst, out);
    e.b.ins().jump(cont, &[]);
}

fn emit_put_by_val(e: &mut Emit, base: VR, idx: VR, val: VR, bc: u32, cont: Block) {
    let bv = e.use_(base);
    let iv = e.use_(idx);
    let vv = e.use_(val);
    let ncm = e.ncm(); let ntag = e.ntag();
    let s1 = e.b.create_block(); let s2 = e.b.create_block(); let s3 = e.b.create_block();
    let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let nc = e.b.ins().band(bv, ncm);
    e.b.ins().brif(nc, slow, &[], s1, &[]);
    e.b.switch_to_block(s1);
    let tb = e.b.ins().band(iv, ntag);
    let ii = e.b.ins().icmp(IntCC::Equal, tb, ntag);
    e.b.ins().brif(ii, s2, &[], slow, &[]);
    e.b.switch_to_block(s2);
    let idx32 = e.b.ins().ireduce(I32, iv);
    let bf = e.b.ins().load(I64, mt(), bv, JSOBJ_BUTTERFLY);
    let len = e.b.ins().load(I32, mt(), bf, -4); // vectorLength
    let inb = e.b.ins().icmp(IntCC::UnsignedLessThan, idx32, len);
    e.b.ins().brif(inb, s3, &[], slow, &[]);
    e.b.switch_to_block(s3);
    let idx64 = e.b.ins().uextend(I64, idx32);
    let sc = e.b.ins().ishl_imm(idx64, 3);
    let addr = e.b.ins().iadd(bf, sc);
    e.b.ins().store(mt(), vv, addr, 0);
    e.b.ins().jump(cont, &[]);
    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go(); let cfr = e.cfr();
    let bc_v = e.b.ins().iconst(I32, bc as i64);
    e.slow_call_void(Slow::PutByVal, &[go, cfr, bv, iv, vv, bc_v]);
    e.check_exc_and_continue(cont);
}

fn emit_get_length(e: &mut Emit, dst: VR, base: VR, bc: u32, cont: Block) {
    // Fast: JSArray-shaped — load butterfly publicLength, box int32.
    let bv = e.use_(base);
    let ncm = e.ncm();
    let nc = e.b.ins().band(bv, ncm);
    let s1 = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let merge = e.b.create_block(); e.b.append_block_param(merge, I64);
    e.b.ins().brif(nc, slow, &[], s1, &[]);
    e.b.switch_to_block(s1);
    let it = e.b.ins().load(I8, mt(), bv, JSCELL_INDEXING);
    let it32 = e.b.ins().uextend(I32, it);
    let has = e.b.ins().band_imm(it32, 0x0e); // IsArray|HasContiguous-ish mask (probe-shape)
    let s2 = e.b.create_block();
    e.b.ins().brif(has, s2, &[], slow, &[]);
    e.b.switch_to_block(s2);
    let bf = e.b.ins().load(I64, mt(), bv, JSOBJ_BUTTERFLY);
    let len = e.b.ins().load(I32, mt(), bf, -8);
    let boxed = e.box_int32(len);
    e.b.ins().jump(merge, &[boxed.into()]);
    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go(); let cfr = e.cfr();
    let bc_v = e.b.ins().iconst(I32, bc as i64);
    let r = e.slow_call_ret(Slow::GetLength, &[go, cfr, bv, bc_v]);
    let post = e.b.create_block();
    e.check_exc_and_continue(post);
    e.b.switch_to_block(post);
    e.b.ins().jump(merge, &[r.into()]);
    e.b.switch_to_block(merge);
    let out = e.b.block_params(merge)[0];
    e.def_(dst, out);
    e.b.ins().jump(cont, &[]);
}

fn emit_add(e: &mut Emit, dst: VR, lhs: Value, rhs: Value, bc: u32, cont: Block) {
    // LLInt: both int32 → 32-bit add w/ overflow check; else slow.
    let ntag = e.ntag();
    let lt = e.b.ins().band(lhs, ntag);
    let rt = e.b.ins().band(rhs, ntag);
    let bothtag = e.b.ins().band(lt, rt);
    let bothint = e.b.ins().icmp(IntCC::Equal, bothtag, ntag);
    let fast = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let merge = e.b.create_block(); e.b.append_block_param(merge, I64);
    e.b.ins().brif(bothint, fast, &[], slow, &[]);
    e.b.switch_to_block(fast);
    let l32 = e.b.ins().ireduce(I32, lhs);
    let r32 = e.b.ins().ireduce(I32, rhs);
    let (sum, ovf) = {
        let r = e.b.ins().sadd_overflow(l32, r32);
        (r.0, r.1)
    };
    let ok = e.b.create_block();
    e.b.ins().brif(ovf, slow, &[], ok, &[]);
    e.b.switch_to_block(ok);
    let boxed = e.box_int32(sum);
    e.b.ins().jump(merge, &[boxed.into()]);
    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go();
    let r = e.slow_call_ret(Slow::Add, &[go, lhs, rhs]);
    let post = e.b.create_block();
    e.check_exc_and_continue(post);
    e.b.switch_to_block(post);
    e.b.ins().jump(merge, &[r.into()]);
    e.b.switch_to_block(merge);
    let out = e.b.block_params(merge)[0];
    e.def_(dst, out);
    e.b.ins().jump(cont, &[]);
}

fn emit_inc(e: &mut Emit, dst: VR, bc: u32, cont: Block) {
    let v = e.use_(dst);
    let one = e.b.ins().iconst(I64, NUMBER_TAG | 1);
    emit_add(e, dst, v, one, bc, cont);
}

fn emit_bitop(e: &mut Emit, kind: u8, dst: VR, lhs: Value, rhs: Value, bc: u32, cont: Block) {
    let ntag = e.ntag();
    let lt = e.b.ins().band(lhs, ntag);
    let rt = e.b.ins().band(rhs, ntag);
    let bt = e.b.ins().band(lt, rt);
    let bi = e.b.ins().icmp(IntCC::Equal, bt, ntag);
    let fast = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let merge = e.b.create_block(); e.b.append_block_param(merge, I64);
    e.b.ins().brif(bi, fast, &[], slow, &[]);
    e.b.switch_to_block(fast);
    let l32 = e.b.ins().ireduce(I32, lhs);
    let r32 = e.b.ins().ireduce(I32, rhs);
    let res32 = match kind {
        0 => e.b.ins().band(l32, r32),
        1 => e.b.ins().bor(l32, r32),
        2 => e.b.ins().bxor(l32, r32),
        3 => { let amt = e.b.ins().band_imm(r32, 31); e.b.ins().sshr(l32, amt) }
        _ => unreachable!(),
    };
    let boxed = e.box_int32(res32);
    e.b.ins().jump(merge, &[boxed.into()]);
    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go();
    let r = e.slow_call_ret(Slow::BitOp, &[go, lhs, rhs]);
    let post = e.b.create_block();
    e.check_exc_and_continue(post);
    e.b.switch_to_block(post);
    e.b.ins().jump(merge, &[r.into()]);
    e.b.switch_to_block(merge);
    let out = e.b.block_params(merge)[0];
    e.def_(dst, out);
    e.b.ins().jump(cont, &[]);
}

fn emit_jless(e: &mut Emit, lhs: VR, rhs: VR, target: Block, fallthru: Block, invert: bool, bc: u32) {
    let lv = e.use_(lhs); let rv = e.use_(rhs);
    let ntag = e.ntag();
    let lt = e.b.ins().band(lv, ntag);
    let rt = e.b.ins().band(rv, ntag);
    let bt = e.b.ins().band(lt, rt);
    let bi = e.b.ins().icmp(IntCC::Equal, bt, ntag);
    let fast = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let decide = e.b.create_block(); e.b.append_block_param(decide, I8);
    e.b.ins().brif(bi, fast, &[], slow, &[]);
    e.b.switch_to_block(fast);
    let l32 = e.b.ins().ireduce(I32, lv);
    let r32 = e.b.ins().ireduce(I32, rv);
    let cmp = e.b.ins().icmp(IntCC::SignedLessThan, l32, r32);
    e.b.ins().jump(decide, &[cmp.into()]);
    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go();
    let r = e.slow_call_ret(Slow::Less, &[go, lv, rv]);
    let r8 = e.b.ins().ireduce(I8, r);
    let post = e.b.create_block();
    e.check_exc_and_continue(post);
    e.b.switch_to_block(post);
    e.b.ins().jump(decide, &[r8.into()]);
    e.b.switch_to_block(decide);
    let c = e.b.block_params(decide)[0];
    if invert {
        e.b.ins().brif(c, fallthru, &[], target, &[]);
    } else {
        e.b.ins().brif(c, target, &[], fallthru, &[]);
    }
}

fn emit_jstricteq(e: &mut Emit, lv: Value, rv: Value, target: Block, fallthru: Block, invert: bool) {
    // LLInt: bitwise-equal → true; both cells → slow; both numbers → slow; else false.
    let beq = e.b.ins().icmp(IntCC::Equal, lv, rv);
    let chk = e.b.create_block();
    let yes = if invert { fallthru } else { target };
    let no = if invert { target } else { fallthru };
    e.b.ins().brif(beq, yes, &[], chk, &[]);
    e.b.switch_to_block(chk);
    // both-cell path → slow
    let ncm = e.ncm();
    let lnc = e.b.ins().band(lv, ncm);
    let rnc = e.b.ins().band(rv, ncm);
    let either = e.b.ins().bor(lnc, rnc);
    let slow = e.b.create_block(); e.b.set_cold_block(slow);
    e.b.ins().brif(either, no, &[], slow, &[]);
    e.b.switch_to_block(slow);
    let go = e.go();
    let r = e.slow_call_ret(Slow::StrictEq, &[go, lv, rv]);
    e.b.ins().brif(r, yes, &[], no, &[]);
}

fn emit_jbool(e: &mut Emit, v: VR, target: Block, fallthru: Block, sense_true: bool) {
    // LLInt: if v is bool (v|1==JS_TRUE) → test low bit; else slow toBoolean.
    let vv = e.use_(v);
    let masked = e.b.ins().bor_imm(vv, 1);
    let k = e.b.ins().iconst(I64, JS_TRUE);
    let isbool = e.b.ins().icmp(IntCC::Equal, masked, k);
    let fast = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let decide = e.b.create_block(); e.b.append_block_param(decide, I64);
    e.b.ins().brif(isbool, fast, &[], slow, &[]);
    e.b.switch_to_block(fast);
    let lo = e.b.ins().band_imm(vv, 1);
    e.b.ins().jump(decide, &[lo.into()]);
    e.b.switch_to_block(slow);
    let go = e.go();
    let r = e.slow_call_ret(Slow::ToBoolean, &[go, vv]);
    e.b.ins().jump(decide, &[r.into()]);
    e.b.switch_to_block(decide);
    let c = e.b.block_params(decide)[0];
    if sense_true {
        e.b.ins().brif(c, target, &[], fallthru, &[]);
    } else {
        e.b.ins().brif(c, fallthru, &[], target, &[]);
    }
}

fn emit_resolve_scope(e: &mut Emit, dst: VR, scope: VR, md_off: i32, bc: u32, cont: Block) {
    // LLInt fast: ClosureVar — walk depth from metadata.localScopeDepth.
    let sv = e.use_(scope);
    let md = e.md();
    let kind = e.b.ins().load(I32, ro(), md, md_off + 0); // resolveType
    let isfast = e.b.ins().icmp_imm(IntCC::UnsignedLessThan, kind, 6);
    let fast = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let merge = e.b.create_block(); e.b.append_block_param(merge, I64);
    e.b.ins().brif(isfast, fast, &[], slow, &[]);
    e.b.switch_to_block(fast);
    // depth-0 approximation: load metadata.constantScope OR pass scope through.
    let cs = e.b.ins().load(I64, ro(), md, md_off + 8);
    let usecs = e.b.create_block(); let usearg = e.b.create_block();
    e.b.ins().brif(cs, usecs, &[], usearg, &[]);
    e.b.switch_to_block(usecs);
    e.b.ins().jump(merge, &[cs.into()]);
    e.b.switch_to_block(usearg);
    e.b.ins().jump(merge, &[sv.into()]);
    e.b.switch_to_block(slow);
    let go = e.go(); let cfr = e.cfr();
    let mo = e.b.ins().iconst(I32, md_off as i64);
    let r = e.slow_call_ret(Slow::ResolveScope, &[go, cfr, mo]);
    e.b.ins().jump(merge, &[r.into()]);
    e.b.switch_to_block(merge);
    let out = e.b.block_params(merge)[0];
    e.def_(dst, out);
    e.b.ins().jump(cont, &[]);
}

fn emit_get_from_scope(e: &mut Emit, dst: VR, scope: VR, md_off: i32, bc: u32, cont: Block) {
    let sv = e.use_(scope);
    let md = e.md();
    let kind = e.b.ins().load(I32, ro(), md, md_off + 0);
    let isfast = e.b.ins().icmp_imm(IntCC::UnsignedLessThan, kind, 6);
    let fast = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    let merge = e.b.create_block(); e.b.append_block_param(merge, I64);
    e.b.ins().brif(isfast, fast, &[], slow, &[]);
    e.b.switch_to_block(fast);
    // operand = metadata.operand → load scope[operand*8 + JSLexicalEnvironment::offsetOfVariables]
    let off = e.b.ins().load(I32, ro(), md, md_off + 16);
    let off64 = e.b.ins().sextend(I64, off);
    let sc = e.b.ins().ishl_imm(off64, 3);
    let addr = e.b.ins().iadd(sv, sc);
    let val = e.b.ins().load(I64, mt(), addr, 64); // offsetOfVariables ≈ 64
    e.b.ins().jump(merge, &[val.into()]);
    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go(); let cfr = e.cfr();
    let mo = e.b.ins().iconst(I32, md_off as i64);
    let r = e.slow_call_ret(Slow::GetFromScope, &[go, cfr, sv, mo]);
    let post = e.b.create_block();
    e.check_exc_and_continue(post);
    e.b.switch_to_block(post);
    e.b.ins().jump(merge, &[r.into()]);
    e.b.switch_to_block(merge);
    let out = e.b.block_params(merge)[0];
    e.def_(dst, out);
    e.b.ins().jump(cont, &[]);
}

fn emit_put_to_scope(e: &mut Emit, scope: VR, val: VR, md_off: i32, bc: u32, cont: Block) {
    let sv = e.use_(scope);
    let vv = e.use_(val);
    let md = e.md();
    let kind = e.b.ins().load(I32, ro(), md, md_off + 0);
    let isfast = e.b.ins().icmp_imm(IntCC::UnsignedLessThan, kind, 6);
    let fast = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    e.b.ins().brif(isfast, fast, &[], slow, &[]);
    e.b.switch_to_block(fast);
    let off = e.b.ins().load(I32, ro(), md, md_off + 16);
    let off64 = e.b.ins().sextend(I64, off);
    let sc = e.b.ins().ishl_imm(off64, 3);
    let addr = e.b.ins().iadd(sv, sc);
    e.b.ins().store(mt(), vv, addr, 64);
    e.b.ins().jump(cont, &[]);
    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go(); let cfr = e.cfr();
    let mo = e.b.ins().iconst(I32, md_off as i64);
    e.slow_call_void(Slow::PutToScope, &[go, cfr, sv, vv, mo]);
    e.check_exc_and_continue(cont);
}

fn emit_call(e: &mut Emit, dst: VR, callee: VR, this: VR, args: &[VR], frame_size: i32, bc: u32, cont: Block) {
    // Write callee frame header below current frame, then dispatch.
    let cfr = e.cfr();
    let callee_v = e.use_(callee);
    let this_v = e.use_(this);
    let nf_off = -(frame_size * 8);
    let nf = e.b.ins().iadd_imm(cfr, nf_off as i64);
    // header
    e.b.ins().store(mt(), callee_v, nf, CFR_CALLEE);
    let argc = e.b.ins().iconst(I32, (args.len() as i64) + 1);
    e.b.ins().store(mt(), argc, nf, CFR_ARGCOUNT);
    e.b.ins().store(mt(), this_v, nf, CFR_THIS);
    for (i, a) in args.iter().enumerate() {
        let av = e.use_(*a);
        e.b.ins().store(mt(), av, nf, CFR_ARG0 + (i as i32) * 8);
    }
    e.stamp_callsite(bc);
    let go = e.go();
    let bc_v = e.b.ins().iconst(I32, bc as i64);
    let argc64 = e.b.ins().uextend(I64, argc);
    let fr = e.slows[&Slow::DispatchCall];
    let call = e.b.ins().call(fr, &[go, cfr, callee_v, this_v, nf, argc64, bc_v]);
    let r = e.b.inst_results(call)[0];
    let post = e.b.create_block();
    e.check_exc_and_continue(post);
    e.b.switch_to_block(post);
    e.def_(dst, r);
    e.b.ins().jump(cont, &[]);
}

fn emit_check_traps(e: &mut Emit, bc: u32, cont: Block) {
    let vm = e.vm();
    let bits = e.b.ins().load(I32, mt(), vm, VM_TRAPS_OFFSET);
    let need = e.b.ins().band_imm(bits, 0xff);
    let slow = e.b.create_block(); e.b.set_cold_block(slow);
    e.b.ins().brif(need, slow, &[], cont, &[]);
    e.b.switch_to_block(slow);
    e.stamp_callsite(bc);
    let go = e.go();
    let r = e.slow_call_ret(Slow::HandleTraps, &[go]);
    let exc_sink = e.exc_sink;
    e.b.ins().brif(r, exc_sink, &[], cont, &[]);
}

fn emit_to_this(e: &mut Emit, cont: Block) {
    let tv = e.use_(0);
    let ncm = e.ncm();
    let nc = e.b.ins().band(tv, ncm);
    let s1 = e.b.create_block(); let slow = e.b.create_block(); e.b.set_cold_block(slow);
    e.b.ins().brif(nc, slow, &[], s1, &[]);
    e.b.switch_to_block(s1);
    let ty = e.b.ins().load(I8, mt(), tv, JSCELL_TYPE);
    let ty32 = e.b.ins().uextend(I32, ty);
    let isobj = e.b.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, ty32, 0x17); // ObjectType
    e.b.ins().brif(isobj, cont, &[], slow, &[]);
    e.b.switch_to_block(slow);
    let go = e.go(); let cfr = e.cfr();
    let r = e.slow_call_ret(Slow::ToThis, &[go, cfr, tv]);
    let post = e.b.create_block();
    e.check_exc_and_continue(post);
    e.b.switch_to_block(post);
    e.def_(0, r);
    e.b.ins().jump(cont, &[]);
}

// ---------------------------------------------------------------------------
// Body driver
// ---------------------------------------------------------------------------

fn emit_body(e: &mut Emit, ops: &[Op]) {
    let mut i = 0;
    while i < ops.len() {
        let op = &ops[i];
        // For ops that need a continuation block, create it lazily; for
        // straight-line ops just emit in current block.
        match op {
            Op::Enter => { /* prologue already done */ }
            Op::LoopHint => {}
            Op::Label(id) => {
                let blk = e.label(*id);
                // Previous op may have been a terminator (Jmp/Ret/J*); check by
                // probing whether the current block already has a terminator.
                let cur = e.b.current_block();
                let needs_jump = match cur {
                    Some(cb) => e.b.func.layout.last_inst(cb)
                        .map(|i| !e.b.func.dfg.insts[i].opcode().is_terminator())
                        .unwrap_or(true),
                    None => false,
                };
                if needs_jump {
                    e.b.ins().jump(blk, &[]);
                }
                e.b.switch_to_block(blk);
            }
            Op::Mov(d, s) => { let v = e.use_(*s); e.def_(*d, v); }
            Op::MovK(d, k) => { let v = e.b.ins().iconst(I64, *k); e.def_(*d, v); }
            Op::NeqNull { dst, src } => {
                // result = jsBoolean( !((src & ~undefinedBit) == null) )
                let sv = e.use_(*src);
                let m = e.b.ins().band_imm(sv, !8i64);
                let isnull = e.b.ins().icmp_imm(IntCC::Equal, m, JS_NULL);
                let n64 = e.b.ins().uextend(I64, isnull);
                let inv = e.b.ins().bxor_imm(n64, 1);
                let r = e.b.ins().bor_imm(inv, JS_FALSE);
                e.def_(*dst, r);
            }
            Op::Jmp(t) => { let blk = e.label(*t); e.b.ins().jump(blk, &[]); }
            Op::Ret(v) => {
                let rv = match v { Some(r) => e.use_(*r), None => e.b.ins().iconst(I64, JS_UNDEFINED) };
                e.b.ins().return_(&[rv]);
            }
            Op::RetK(k) => { let rv = e.b.ins().iconst(I64, *k); e.b.ins().return_(&[rv]); }
            Op::ToThis => { let c = e.b.create_block(); emit_to_this(e, c); e.b.switch_to_block(c); }
            Op::CheckTraps { bc } => { let c = e.b.create_block(); emit_check_traps(e, *bc, c); e.b.switch_to_block(c); }
            Op::GetById { dst, base, md_off, ident, bc } => {
                let c = e.b.create_block(); emit_get_by_id(e, *dst, *base, *md_off, *ident, *bc, c); e.b.switch_to_block(c);
            }
            Op::PutById { base, val, md_off, ident, bc } => {
                let c = e.b.create_block(); emit_put_by_id(e, *base, *val, *md_off, *ident, *bc, c); e.b.switch_to_block(c);
            }
            Op::GetByVal { dst, base, idx, bc } => {
                let c = e.b.create_block(); emit_get_by_val(e, *dst, *base, *idx, *bc, c); e.b.switch_to_block(c);
            }
            Op::PutByVal { base, idx, val, bc } => {
                let c = e.b.create_block(); emit_put_by_val(e, *base, *idx, *val, *bc, c); e.b.switch_to_block(c);
            }
            Op::GetLength { dst, base, bc } => {
                let c = e.b.create_block(); emit_get_length(e, *dst, *base, *bc, c); e.b.switch_to_block(c);
            }
            Op::Add { dst, lhs, rhs, bc } => {
                let l = e.use_(*lhs); let r = e.use_(*rhs);
                let c = e.b.create_block(); emit_add(e, *dst, l, r, *bc, c); e.b.switch_to_block(c);
            }
            Op::AddK { dst, lhs, k, bc } => {
                let l = e.use_(*lhs);
                let r = e.b.ins().iconst(I64, NUMBER_TAG | (*k as u32 as i64));
                let c = e.b.create_block(); emit_add(e, *dst, l, r, *bc, c); e.b.switch_to_block(c);
            }
            Op::Inc { dst, bc } => { let c = e.b.create_block(); emit_inc(e, *dst, *bc, c); e.b.switch_to_block(c); }
            Op::BitOp { kind, dst, lhs, rhs, bc } => {
                let l = e.use_(*lhs); let r = e.use_(*rhs);
                let c = e.b.create_block(); emit_bitop(e, *kind, *dst, l, r, *bc, c); e.b.switch_to_block(c);
            }
            Op::BitOpK { kind, dst, lhs, k, bc } => {
                let l = e.use_(*lhs);
                let r = e.b.ins().iconst(I64, NUMBER_TAG | (*k as u32 as i64));
                let c = e.b.create_block(); emit_bitop(e, *kind, *dst, l, r, *bc, c); e.b.switch_to_block(c);
            }
            Op::ResolveScope { dst, scope, md_off, bc } => {
                let c = e.b.create_block(); emit_resolve_scope(e, *dst, *scope, *md_off, *bc, c); e.b.switch_to_block(c);
            }
            Op::GetFromScope { dst, scope, md_off, bc } => {
                let c = e.b.create_block(); emit_get_from_scope(e, *dst, *scope, *md_off, *bc, c); e.b.switch_to_block(c);
            }
            Op::PutToScope { scope, val, md_off, bc } => {
                let c = e.b.create_block(); emit_put_to_scope(e, *scope, *val, *md_off, *bc, c); e.b.switch_to_block(c);
            }
            Op::Call { dst, callee, this, args, frame_size, bc } => {
                let c = e.b.create_block(); emit_call(e, *dst, *callee, *this, args, *frame_size, *bc, c); e.b.switch_to_block(c);
            }
            Op::JTrue { cond, target } => { let t = e.label(*target); let f = e.b.create_block(); emit_jbool(e, *cond, t, f, true); e.b.switch_to_block(f); }
            Op::JFalse { cond, target } => { let t = e.label(*target); let f = e.b.create_block(); emit_jbool(e, *cond, t, f, false); e.b.switch_to_block(f); }
            Op::JEqNull { v, target } => {
                let vv = e.use_(*v);
                let m = e.b.ins().band_imm(vv, !8i64);
                let isnull = e.b.ins().icmp_imm(IntCC::Equal, m, JS_NULL);
                let t = e.label(*target); let f = e.b.create_block();
                e.b.ins().brif(isnull, t, &[], f, &[]); e.b.switch_to_block(f);
            }
            Op::JNEqNull { v, target } => {
                let vv = e.use_(*v);
                let m = e.b.ins().band_imm(vv, !8i64);
                let isnull = e.b.ins().icmp_imm(IntCC::Equal, m, JS_NULL);
                let t = e.label(*target); let f = e.b.create_block();
                e.b.ins().brif(isnull, f, &[], t, &[]); e.b.switch_to_block(f);
            }
            Op::JStrictEq { lhs, rhs, target } => {
                let l = e.use_(*lhs); let r = e.use_(*rhs);
                let t = e.label(*target); let f = e.b.create_block();
                emit_jstricteq(e, l, r, t, f, false); e.b.switch_to_block(f);
            }
            Op::JNStrictEq { lhs, rhs, target } => {
                let l = e.use_(*lhs); let r = e.use_(*rhs);
                let t = e.label(*target); let f = e.b.create_block();
                emit_jstricteq(e, l, r, t, f, true); e.b.switch_to_block(f);
            }
            Op::JStrictEqK { lhs, k, target } => {
                let l = e.use_(*lhs); let r = e.b.ins().iconst(I64, *k);
                let t = e.label(*target); let f = e.b.create_block();
                emit_jstricteq(e, l, r, t, f, false); e.b.switch_to_block(f);
            }
            Op::JNStrictEqK { lhs, k, target } => {
                let l = e.use_(*lhs); let r = e.b.ins().iconst(I64, *k);
                let t = e.label(*target); let f = e.b.create_block();
                emit_jstricteq(e, l, r, t, f, true); e.b.switch_to_block(f);
            }
            Op::JLess { lhs, rhs, target, bc } => {
                let t = e.label(*target); let f = e.b.create_block();
                emit_jless(e, *lhs, *rhs, t, f, false, *bc); e.b.switch_to_block(f);
            }
            Op::JNLess { lhs, rhs, target, bc } => {
                let t = e.label(*target); let f = e.b.create_block();
                emit_jless(e, *lhs, *rhs, t, f, true, *bc); e.b.switch_to_block(f);
            }
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Slow-path import declarations
// ---------------------------------------------------------------------------

/// Returns (params, has_result, name). DispatchCall is always SystemV with a
/// real return; everything else is preserve_all in pa-mode (result via
/// trailing out-pointer) or SystemV in sv-mode (direct result).
fn slow_sig(s: Slow) -> (Vec<AbiParam>, bool, &'static str) {
    use AbiParam as P;
    let p64 = || P::new(I64);
    let p32 = || P::new(I32);
    match s {
        Slow::GetById => (vec![p64(), p64(), p64(), p32(), p32()], true, "aotSlowGetById"),
        Slow::PutById => (vec![p64(), p64(), p64(), p64(), p32(), p32()], false, "aotSlowPutById"),
        Slow::GetByVal => (vec![p64(), p64(), p64(), p64(), p32()], true, "aotSlowGetByVal"),
        Slow::PutByVal => (vec![p64(), p64(), p64(), p64(), p64(), p32()], false, "aotSlowPutByVal"),
        Slow::GetLength => (vec![p64(), p64(), p64(), p32()], true, "aotSlowGetLength"),
        Slow::Add => (vec![p64(), p64(), p64()], true, "aotSlowAdd"),
        Slow::Less => (vec![p64(), p64(), p64()], true, "aotSlowLess"),
        Slow::Inc => (vec![p64(), p64()], true, "aotSlowInc"),
        Slow::BitOp => (vec![p64(), p64(), p64()], true, "aotSlowBitOp"),
        Slow::ToThis => (vec![p64(), p64(), p64()], true, "aotSlowToThis"),
        Slow::ResolveScope => (vec![p64(), p64(), p32()], true, "aotSlowResolveScope"),
        Slow::GetFromScope => (vec![p64(), p64(), p64(), p32()], true, "aotSlowGetFromScope"),
        Slow::PutToScope => (vec![p64(), p64(), p64(), p64(), p32()], false, "aotSlowPutToScope"),
        Slow::DispatchCall => (vec![p64(), p64(), p64(), p64(), p64(), p64(), p32()], true, "aotDispatchCall"),
        Slow::HandleTraps => (vec![p64()], true, "aotHandleTraps"),
        Slow::StrictEq => (vec![p64(), p64(), p64()], true, "aotSlowStrictEq"),
        Slow::ToBoolean => (vec![p64(), p64()], true, "aotSlowToBoolean"),
    }
}

const ALL_SLOWS: &[Slow] = &[
    Slow::GetById, Slow::PutById, Slow::GetByVal, Slow::PutByVal, Slow::GetLength,
    Slow::Add, Slow::Less, Slow::Inc, Slow::BitOp, Slow::ToThis, Slow::ResolveScope,
    Slow::GetFromScope, Slow::PutToScope, Slow::DispatchCall, Slow::HandleTraps,
    Slow::StrictEq, Slow::ToBoolean,
];

// ---------------------------------------------------------------------------
// The 5 bodies (opcode sequences derived from real C-AOT gen comments)
// ---------------------------------------------------------------------------

fn jsint(k: i32) -> i64 { NUMBER_TAG | (k as u32 as i64) }

fn body_arehookinputsequal() -> Body {
    // From newp/fl-base-render-microbench fn#2. 19 bytecode ops.
    use Op::*;
    Body { name: "areHookInputsEqual", n_args: 2, n_locals: 4, ops: vec![
        Enter,
        JNStrictEqK { lhs: 2, k: JS_NULL, target: 7 },
        RetK(JS_FALSE),
        Label(7),
        MovK(-2, jsint(0)),
        GetLength { dst: -3, base: 2, bc: 10 },
        JNLess { lhs: -2, rhs: -3, target: 69, bc: 15 },
        GetLength { dst: -3, base: 1, bc: 19 },
        JNLess { lhs: -2, rhs: -3, target: 69, bc: 24 },
        Label(28),
        LoopHint,
        CheckTraps { bc: 29 },
        GetByVal { dst: -3, base: 1, idx: -2, bc: 30 },
        GetByVal { dst: -4, base: 2, idx: -2, bc: 36 },
        JStrictEq { lhs: -3, rhs: -4, target: 48 },
        RetK(JS_FALSE),
        Label(48),
        Inc { dst: -2, bc: 48 },
        GetLength { dst: -3, base: 2, bc: 51 },
        JNLess { lhs: -2, rhs: -3, target: 69, bc: 56 },
        GetLength { dst: -3, base: 1, bc: 60 },
        JLess { lhs: -2, rhs: -3, target: 28, bc: 65 },
        Label(69),
        RetK(JS_TRUE),
    ]}
}

fn body_updateworkinprogresshook() -> Body {
    // From fn1inv/c-dump fn#48362, truncated to first 41 ops (representative
    // resolve_scope/get_from_scope/get_by_id/put_by_id chain + 2 calls).
    use Op::*;
    Body { name: "updateWorkInProgressHook", n_args: 0, n_locals: 8, ops: vec![
        Enter,
        ResolveScope { dst: -5, scope: -1, md_off: 200, bc: 1 },
        GetFromScope { dst: -5, scope: -5, md_off: 232, bc: 8 },
        JNStrictEqK { lhs: -5, k: JS_NULL, target: 60 },
        ResolveScope { dst: -6, scope: -1, md_off: 248, bc: 21 },
        GetFromScope { dst: -6, scope: -6, md_off: 280, bc: 28 },
        GetById { dst: -2, base: -6, md_off: 312, ident: 0, bc: 37 },
        JStrictEqK { lhs: -2, k: JS_NULL, target: 55 },
        GetById { dst: -2, base: -2, md_off: 328, ident: 1, bc: 47 },
        Jmp(58),
        Label(55),
        MovK(-2, JS_NULL),
        Label(58),
        Jmp(82),
        Label(60),
        ResolveScope { dst: -6, scope: -1, md_off: 344, bc: 60 },
        GetFromScope { dst: -6, scope: -6, md_off: 376, bc: 67 },
        GetById { dst: -2, base: -6, md_off: 408, ident: 2, bc: 76 },
        Label(82),
        ResolveScope { dst: -6, scope: -1, md_off: 424, bc: 82 },
        GetFromScope { dst: -5, scope: -6, md_off: 456, bc: 89 },
        JNStrictEqK { lhs: -5, k: JS_NULL, target: 126 },
        ResolveScope { dst: -6, scope: -1, md_off: 488, bc: 102 },
        GetFromScope { dst: -6, scope: -6, md_off: 520, bc: 109 },
        GetById { dst: -3, base: -6, md_off: 552, ident: 3, bc: 118 },
        Jmp(148),
        Label(126),
        ResolveScope { dst: -6, scope: -1, md_off: 568, bc: 126 },
        GetFromScope { dst: -6, scope: -6, md_off: 600, bc: 133 },
        GetById { dst: -3, base: -6, md_off: 632, ident: 4, bc: 142 },
        Label(148),
        ResolveScope { dst: -6, scope: -1, md_off: 648, bc: 152 },
        PutToScope { scope: -6, val: -2, md_off: 680, bc: 159 },
        ResolveScope { dst: -6, scope: -1, md_off: 712, bc: 167 },
        PutToScope { scope: -6, val: -3, md_off: 744, bc: 174 },
        JStrictEqK { lhs: -3, k: JS_NULL, target: 184 },
        // call path
        ResolveScope { dst: -7, scope: -1, md_off: 776, bc: 188 },
        GetFromScope { dst: -7, scope: -7, md_off: 808, bc: 195 },
        GetById { dst: -8, base: -7, md_off: 840, ident: 5, bc: 204 },
        Call { dst: -4, callee: -8, this: -7, args: vec![-3], frame_size: 16, bc: 249 },
        Call { dst: -4, callee: -4, this: -7, args: vec![], frame_size: 16, bc: 256 },
        Label(184),
        GetById { dst: -4, base: -2, md_off: 856, ident: 6, bc: 351 },
        PutById { base: -3, val: -4, md_off: 880, ident: 6, bc: 357 },
        GetById { dst: -4, base: -2, md_off: 904, ident: 7, bc: 379 },
        PutById { base: -3, val: -4, md_off: 928, ident: 7, bc: 385 },
        GetById { dst: -4, base: -2, md_off: 952, ident: 8, bc: 407 },
        PutById { base: -3, val: -4, md_off: 976, ident: 8, bc: 413 },
        MovK(-4, JS_NULL),
        PutById { base: -3, val: -4, md_off: 1000, ident: 9, bc: 447 },
        ResolveScope { dst: -6, scope: -1, md_off: 1024, bc: 558 },
        GetFromScope { dst: -4, scope: -6, md_off: 1056, bc: 565 },
        Ret(Some(-4)),
    ]}
}

fn body_scheduler_schedule() -> Body {
    // From rollint/fl-richards fn#15. 28 bytecode ops.
    use Op::*;
    Body { name: "scheduler_schedule", n_args: 0, n_locals: 4, ops: vec![
        Enter,
        ToThis,
        Mov(-2, 0),
        GetById { dst: -3, base: 0, md_off: 200, ident: 0, bc: 9 },
        PutById { base: -2, val: -3, md_off: 104, ident: 1, bc: 15 },
        GetById { dst: -2, base: 0, md_off: 216, ident: 1, bc: 21 },
        JEqNull { v: -2, target: 138 },
        Label(30),
        LoopHint,
        CheckTraps { bc: 31 },
        GetById { dst: -4, base: 0, md_off: 232, ident: 1, bc: 32 },
        GetById { dst: -2, base: -4, md_off: 248, ident: 2, bc: 38 },
        Call { dst: -2, callee: -2, this: -4, args: vec![], frame_size: 10, bc: 44 },
        JFalse { cond: -2, target: 77 },
        Mov(-2, 0),
        GetById { dst: -4, base: 0, md_off: 264, ident: 1, bc: 57 },
        GetById { dst: -3, base: -4, md_off: 280, ident: 3, bc: 63 },
        PutById { base: -2, val: -3, md_off: 128, ident: 1, bc: 69 },
        Jmp(126),
        Label(77),
        Mov(-2, 0),
        GetById { dst: -4, base: 0, md_off: 296, ident: 1, bc: 80 },
        GetById { dst: -3, base: -4, md_off: 312, ident: 4, bc: 86 },
        PutById { base: -2, val: -3, md_off: 152, ident: 5, bc: 92 },
        Mov(-2, 0),
        GetById { dst: -4, base: 0, md_off: 328, ident: 1, bc: 101 },
        GetById { dst: -3, base: -4, md_off: 344, ident: 6, bc: 107 },
        Call { dst: -3, callee: -3, this: -4, args: vec![], frame_size: 10, bc: 113 },
        PutById { base: -2, val: -3, md_off: 176, ident: 1, bc: 120 },
        Label(126),
        GetById { dst: -2, base: 0, md_off: 360, ident: 1, bc: 126 },
        NeqNull { dst: -2, src: -2 },
        JTrue { cond: -2, target: 30 },
        Label(138),
        Ret(None),
    ]}
}

fn body_closurevar_fn5() -> Body {
    // From rollint/fl-gap-closurevar fn#5. 23 bytecode ops.
    use Op::*;
    Body { name: "closurevar_fn5", n_args: 1, n_locals: 14, ops: vec![
        Enter,
        // op_enter scope seed: loc1 = callee->scope()
        // (modeled in prologue via -1 = load cfr.callee.scope below)
        MovK(-3, jsint(0)),
        MovK(-4, jsint(0)),
        JNLess { lhs: -4, rhs: 1, target: 85, bc: 10 },
        Label(14),
        LoopHint,
        CheckTraps { bc: 15 },
        Mov(-5, -3),
        ResolveScope { dst: -14, scope: -1, md_off: 200, bc: 19 },
        GetFromScope { dst: -6, scope: -14, md_off: 232, bc: 26 },
        Mov(-13, -4),
        AddK { dst: -12, lhs: -4, k: 1, bc: 38 },
        AddK { dst: -11, lhs: -4, k: 2, bc: 44 },
        AddK { dst: -10, lhs: -4, k: 3, bc: 50 },
        AddK { dst: -9, lhs: -4, k: 4, bc: 56 },
        Call { dst: -6, callee: -6, this: -14, args: vec![-13, -12, -11, -10, -9], frame_size: 20, bc: 62 },
        Add { dst: -5, lhs: -5, rhs: -6, bc: 69 },
        Mov(-3, -5),
        Inc { dst: -4, bc: 78 },
        JLess { lhs: -4, rhs: 1, target: 14, bc: 81 },
        Label(85),
        ResolveScope { dst: -5, scope: -1, md_off: 216, bc: 85 },
        GetFromScope { dst: -6, scope: -5, md_off: 256, bc: 92 },
        Add { dst: -7, lhs: -3, rhs: -6, bc: 101 },
        Ret(Some(-7)),
    ]}
}

fn body_typedarray_fn1() -> Body {
    // From rollint/fl-gap-typedarray fn#1. 33 bytecode ops.
    use Op::*;
    Body { name: "typedarray_fn1", n_args: 3, n_locals: 12, ops: vec![
        Enter,
        GetLength { dst: -9, base: 2, bc: 1 },
        BitOpK { kind: 1, dst: -5, lhs: -9, k: 0, bc: 6 },
        MovK(-6, jsint(0)),
        MovK(-2, jsint(0)),
        MovK(-4, jsint(0)),
        JNLess { lhs: -4, rhs: -5, target: 174, bc: 21 },
        Label(25),
        LoopHint,
        CheckTraps { bc: 26 },
        ResolveScope { dst: -12, scope: -1, md_off: 232, bc: 27 },
        GetFromScope { dst: -9, scope: -12, md_off: 248, bc: 34 },
        Mov(-11, 1),
        Mov(-10, -6),
        Call { dst: -8, callee: -9, this: -12, args: vec![-11, -10], frame_size: 18, bc: 49 },
        BitOpK { kind: 0, dst: -9, lhs: -8, k: 511, bc: 56 },
        GetByVal { dst: -7, base: 3, idx: -9, bc: 62 },
        BitOpK { kind: 3, dst: -9, lhs: -6, k: 3, bc: 68 },
        AddK { dst: -9, lhs: -9, k: 3, bc: 73 },
        GetByVal { dst: -3, base: 1, idx: -9, bc: 79 },
        Mov(-9, 2),
        Mov(-10, -4),
        BitOp { kind: 2, dst: -11, lhs: -8, rhs: -7, bc: 91 },
        BitOp { kind: 2, dst: -11, lhs: -11, rhs: -3, bc: 97 },
        BitOp { kind: 2, dst: -11, lhs: -11, rhs: -2, bc: 103 },
        BitOpK { kind: 0, dst: -11, lhs: -11, k: 255, bc: 109 },
        PutByVal { base: -9, idx: -10, val: -11, bc: 115 },
        GetByVal { dst: -9, base: 2, idx: -4, bc: 121 },
        Add { dst: -10, lhs: -2, rhs: -9, bc: 127 },
        BitOpK { kind: 1, dst: -2, lhs: -10, k: 0, bc: 133 },
        BitOpK { kind: 0, dst: -9, lhs: -7, k: 15, bc: 139 },
        Add { dst: -10, lhs: -6, rhs: -9, bc: 145 },
        AddK { dst: -10, lhs: -10, k: 1, bc: 151 },
        BitOpK { kind: 1, dst: -6, lhs: -10, k: 0, bc: 157 },
        Inc { dst: -4, bc: 163 },
        JLess { lhs: -4, rhs: -5, target: 25, bc: 166 },
        Label(174),
        Ret(Some(-2)),
    ]}
}

// ---------------------------------------------------------------------------
// Build one body into an ObjectModule function + dump artifacts
// ---------------------------------------------------------------------------

struct BodyResult {
    name: String,
    n_ops: usize,
    clif_instrs: usize,
    clif_blocks: usize,
    text_bytes: usize,
    mach_instrs: usize,
}

fn build_body(
    module: &mut ObjectModule,
    fbctx: &mut FunctionBuilderContext,
    slow_ids: &HashMap<Slow, cranelift_module::FuncId>,
    body: &Body,
    outdir: &PathBuf,
    slow_cc_systemv: bool,
) -> Result<BodyResult> {
    let fe_cfg: TargetFrontendConfig = module.isa().frontend_config();
    let mut sig = module.make_signature();
    sig.call_conv = CallConv::SystemV;
    sig.params.push(AbiParam::new(I64)); // globalObject
    sig.params.push(AbiParam::new(I64)); // callFrame
    sig.returns.push(AbiParam::new(I64));
    let fid = module.declare_function(&format!("aotgen_{}", body.name), Linkage::Export, &sig)?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    ctx.func.name = UserFuncName::user(0, fid.as_u32());
    ctx.set_disasm(true);

    {
        let mut b = FunctionBuilder::new(&mut ctx.func, fbctx);

        // Declare pinned vars + locals.
        let go = b.declare_var(I64);
        let cfr = b.declare_var(I64);
        let md = b.declare_var(I64);
        let vm = b.declare_var(I64);
        let ntag = b.declare_var(I64);
        let ncm = b.declare_var(I64);
        let vthis = b.declare_var(I64);
        let args: Vec<Variable> = (0..body.n_args).map(|_| b.declare_var(I64)).collect();
        let locs: Vec<Variable> = (0..body.n_locals).map(|_| b.declare_var(I64)).collect();

        // Import slow-path FuncRefs into this function.
        let mut slows = HashMap::new();
        for s in ALL_SLOWS {
            let fr = module.declare_func_in_func(slow_ids[s], b.func);
            slows.insert(*s, fr);
        }

        // Stack slot for preserve_all out-param.
        let out_slot = b.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 8, 3,
        ));

        // Entry block / prologue.
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        let exc_sink = b.create_block();
        b.set_cold_block(exc_sink);
        b.switch_to_block(entry);
        let p_go = b.block_params(entry)[0];
        let p_cfr = b.block_params(entry)[1];
        b.def_var(go, p_go);
        b.def_var(cfr, p_cfr);
        // vm = go->m_vm
        let vmv = b.ins().load(I64, MemFlagsData::trusted().with_readonly(), p_go, GO_VM_OFFSET);
        b.def_var(vm, vmv);
        // codeBlock = cfr[CodeBlock]; mdBase = codeBlock->metadataTable()
        let cb = b.ins().load(I64, MemFlagsData::trusted().with_readonly(), p_cfr, CFR_CODEBLOCK);
        let mdv = b.ins().load(I64, MemFlagsData::trusted().with_readonly(), cb, CB_METADATA);
        b.def_var(md, mdv);
        // tag constants
        let ntv = b.ins().iconst(I64, NUMBER_TAG);
        b.def_var(ntag, ntv);
        let ncv = b.ins().iconst(I64, NOT_CELL_MASK);
        b.def_var(ncm, ncv);
        // this + args
        let tv = b.ins().load(I64, MemFlagsData::trusted(), p_cfr, CFR_THIS);
        b.def_var(vthis, tv);
        for (i, a) in args.iter().enumerate() {
            let av = b.ins().load(I64, MemFlagsData::trusted(), p_cfr, CFR_ARG0 + 8 * i as i32);
            b.def_var(*a, av);
        }
        // locals init = jsUndefined, except loc1 = scope (callee->scope)
        let und = b.ins().iconst(I64, JS_UNDEFINED);
        let callee = b.ins().load(I64, MemFlagsData::trusted(), p_cfr, CFR_CALLEE);
        let scope = b.ins().load(I64, MemFlagsData::trusted(), callee, JSCALLEE_SCOPE);
        for (i, l) in locs.iter().enumerate() {
            if i == 0 { b.def_var(*l, scope); } else { b.def_var(*l, und); }
        }

        let mut e = Emit {
            b: &mut b, go, cfr, md, vm, ntag, ncm, vthis, args, locs,
            labels: HashMap::new(), exc_sink, slows, out_slot,
            pa: !slow_cc_systemv,
        };

        emit_body(&mut e, &body.ops);

        // exc_sink: return undefined
        b.switch_to_block(exc_sink);
        let u = b.ins().iconst(I64, JS_UNDEFINED);
        b.ins().return_(&[u]);

        b.seal_all_blocks();
        b.finalize(fe_cfg);
    }

    // Count CLIF instrs/blocks pre-opt.
    let mut clif_instrs = 0usize;
    let mut clif_blocks = 0usize;
    for blk in ctx.func.layout.blocks() {
        clif_blocks += 1;
        for _ in ctx.func.layout.block_insts(blk) { clif_instrs += 1; }
    }
    let clif_text = format!("{}", ctx.func.display());
    fs::write(outdir.join(format!("{}.clif", body.name)), &clif_text)?;

    // Compile via module (also leaves ctx.compiled_code populated).
    module.define_function(fid, &mut ctx)?;

    let cc = ctx.compiled_code().unwrap();
    let text_bytes = cc.code_buffer().len();
    let vcode = cc.vcode.clone().unwrap_or_default();
    let mach_instrs = vcode.lines().filter(|l| {
        let t = l.trim_start();
        !t.is_empty() && !t.starts_with("block") && !t.starts_with(';') && !t.starts_with("VCode")
            && !t.starts_with("Disasm") && !t.ends_with(':')
    }).count();
    fs::write(outdir.join(format!("{}.vcode", body.name)), &vcode)?;

    let _ = slow_cc_systemv;

    Ok(BodyResult {
        name: body.name.to_string(),
        n_ops: body.ops.iter().filter(|o| !matches!(o, Op::Label(_) | Op::LoopHint)).count(),
        clif_instrs, clif_blocks, text_bytes, mach_instrs,
    })
}

fn make_module(opt_level: &str) -> Result<ObjectModule> {
    let mut flags = settings::builder();
    flags.set("opt_level", opt_level)?;
    flags.set("enable_verifier", "true")?;
    flags.set("is_pic", "true")?;
    flags.set("preserve_frame_pointers", "true")?;
    let isa_builder = cranelift_native::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
    let isa = isa_builder.finish(settings::Flags::new(flags))?;
    let ob = ObjectBuilder::new(isa, "clprobe", cranelift_module::default_libcall_names())?;
    Ok(ObjectModule::new(ob))
}

fn declare_slows(module: &mut ObjectModule, preserve_all: bool) -> Result<HashMap<Slow, cranelift_module::FuncId>> {
    let mut m = HashMap::new();
    for s in ALL_SLOWS {
        let (mut params, has_ret, name) = slow_sig(*s);
        let mut sig = module.make_signature();
        if matches!(s, Slow::DispatchCall) {
            // op_call dispatch is always a real SystemV call (re-enters JS).
            sig.call_conv = CallConv::SystemV;
            sig.params = params;
            sig.returns = vec![AbiParam::new(I64)];
        } else if preserve_all {
            sig.call_conv = CallConv::PreserveAll;
            if has_ret { params.push(AbiParam::new(I64)); } // out-pointer
            sig.params = params;
            // no returns
        } else {
            sig.call_conv = CallConv::SystemV;
            sig.params = params;
            if has_ret { sig.returns = vec![AbiParam::new(I64)]; }
        }
        let fid = module.declare_function(name, Linkage::Import, &sig)?;
        m.insert(*s, fid);
    }
    Ok(m)
}

fn run_variant(outdir: &PathBuf, tag: &str, preserve_all: bool) -> Result<Vec<BodyResult>> {
    let dir = outdir.join(tag);
    fs::create_dir_all(&dir)?;
    let mut module = make_module("speed")?;
    let slow_ids = declare_slows(&mut module, preserve_all)?;
    let mut fbctx = FunctionBuilderContext::new();
    let slow_cc_systemv = !preserve_all;
    let bodies = vec![
        body_arehookinputsequal(),
        body_updateworkinprogresshook(),
        body_scheduler_schedule(),
        body_closurevar_fn5(),
        body_typedarray_fn1(),
    ];
    let mut results = vec![];
    for body in &bodies {
        let r = build_body(&mut module, &mut fbctx, &slow_ids, body, &dir, slow_cc_systemv)?;
        results.push(r);
    }
    let product = module.finish();
    let obj = product.emit()?;
    fs::write(dir.join("clprobe_all.o"), &obj)?;
    // Also write per-body .o by re-building each into its own module.
    for body in &bodies {
        let mut m1 = make_module("speed")?;
        let s1 = declare_slows(&mut m1, preserve_all)?;
        let mut fbc1 = FunctionBuilderContext::new();
        let _ = build_body(&mut m1, &mut fbc1, &s1, body, &dir, slow_cc_systemv)?;
        let p1 = m1.finish().emit()?;
        fs::write(dir.join(format!("{}.o", body.name)), &p1)?;
    }
    Ok(results)
}

fn main() -> Result<()> {
    let outdir = PathBuf::from(
        std::env::var("CLPROBE_OUT").unwrap_or_else(|_| "/root/src/bun/build/scratch/asmprobe/B-cranelift/out".into()),
    );
    fs::create_dir_all(&outdir)?;

    let r_pa = run_variant(&outdir, "preserve_all", true)?;
    let r_sv = run_variant(&outdir, "systemv", false)?;

    let mut summary = String::new();
    summary.push_str("# clprobe summary (opt_level=speed)\n");
    summary.push_str("# variant=preserve_all: slow paths use CallConv::PreserveAll (DispatchCall stays SystemV)\n");
    summary.push_str("# variant=systemv:      slow paths use CallConv::SystemV (worst case)\n\n");
    for (tag, rs) in [("preserve_all", &r_pa), ("systemv", &r_sv)] {
        summary.push_str(&format!("## {tag}\n"));
        summary.push_str(&format!("{:<28} {:>6} {:>10} {:>10} {:>11} {:>11}\n",
            "body", "ops", "clif-inst", "clif-blks", "mach-inst", ".text bytes"));
        for r in rs {
            summary.push_str(&format!("{:<28} {:>6} {:>10} {:>10} {:>11} {:>11}\n",
                r.name, r.n_ops, r.clif_instrs, r.clif_blocks, r.mach_instrs, r.text_bytes));
        }
        summary.push('\n');
    }
    fs::write(outdir.join("summary.txt"), &summary)?;
    println!("{summary}");
    println!("artifacts → {}", outdir.display());
    Ok(())
}
