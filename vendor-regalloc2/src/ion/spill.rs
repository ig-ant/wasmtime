/*
 * This file was initially derived from the files
 * `js/src/jit/BacktrackingAllocator.h` and
 * `js/src/jit/BacktrackingAllocator.cpp` in Mozilla Firefox, and was
 * originally licensed under the Mozilla Public License 2.0. We
 * subsequently relicensed it to Apache-2.0 WITH LLVM-exception (see
 * https://github.com/bytecodealliance/regalloc2/issues/7).
 *
 * Since the initial port, the design has been substantially evolved
 * and optimized.
 */

//! Spillslot allocation.

use super::{
    AllocRegResult, Env, LiveRangeKey, PRegIndex, RegTraversalIter, SpillSetIndex, SpillSlotData,
    SpillSlotIndex,
};
use crate::{Allocation, Function, SpillSlot};

impl<'a, F: Function> Env<'a, F> {
    pub fn try_allocating_regs_for_spilled_bundles(&mut self) {
        trace!("allocating regs for spilled bundles");
        let mut scratch = core::mem::take(&mut self.ctx.scratch_conflicts);
        for i in 0..self.ctx.spilled_bundles.len() {
            let bundle = self.ctx.spilled_bundles[i]; // don't borrow self

            if self.ctx.bundles[bundle].ranges.is_empty() {
                continue;
            }

            let class = self.ctx.spillsets[self.ctx.bundles[bundle].spillset].class;
            let hint = self.ctx.spillsets[self.ctx.bundles[bundle].spillset]
                .hint
                .as_valid();

            // This may be an empty-range bundle whose ranges are not
            // sorted; sort all range-lists again here.
            self.ctx.bundles[bundle]
                .ranges
                .sort_unstable_by_key(|entry| entry.range.from);

            let mut success = false;
            self.ctx.output.stats.spill_bundle_reg_probes += 1;
            let limit = self.bundles[bundle].limit.map(|l| l as usize);
            for preg in RegTraversalIter::new(self.env, class, None, hint, bundle.index(), limit) {
                trace!("trying bundle {:?} to preg {:?}", bundle, preg);
                let preg_idx = PRegIndex::new(preg.index());
                if let AllocRegResult::Allocated(_) =
                    self.try_to_allocate_bundle_to_reg(bundle, preg_idx, None, &mut scratch)
                {
                    self.ctx.output.stats.spill_bundle_reg_success += 1;
                    success = true;
                    break;
                }
            }

            if !success {
                trace!(
                    "spilling bundle {:?}: marking spillset {:?} as required",
                    bundle,
                    self.ctx.bundles[bundle].spillset
                );
                self.ctx.spillsets[self.ctx.bundles[bundle].spillset].required = true;
            }
        }
        self.ctx.scratch_conflicts = scratch;
    }

    pub fn spillslot_can_fit_spillset(
        &mut self,
        spillslot: SpillSlotIndex,
        spillset: SpillSetIndex,
    ) -> bool {
        !self.ctx.spillslots[spillslot.index()]
            .ranges
            .btree
            .contains_key(&LiveRangeKey::from_range(
                &self.ctx.spillsets[spillset].range,
            ))
    }

    pub fn allocate_spillset_to_spillslot(
        &mut self,
        spillset: SpillSetIndex,
        spillslot: SpillSlotIndex,
    ) {
        self.ctx.spillsets[spillset].slot = spillslot;

        let res = self.ctx.spillslots[spillslot.index()].ranges.btree.insert(
            LiveRangeKey::from_range(&self.ctx.spillsets[spillset].range),
            spillset,
        );

        debug_assert!(res.is_none());
    }

    pub fn allocate_spillslots(&mut self) {
        const MAX_ATTEMPTS: usize = 10;

        let (mut n_req, mut n_fixed) = (0usize, 0usize);
        for spillset in 0..self.ctx.spillsets.len() {
            trace!("allocate spillslot: {}", spillset);
            let spillset = SpillSetIndex::new(spillset);
            if !self.ctx.spillsets[spillset].required {
                continue;
            }
            n_req += 1;
            // Embedder-fixed slot: the spillset's home is already
            // decided (e.g. an interpreter frame local at
            // `[fp − r×8]`). Two spillsets can name the SAME fixed
            // slot (a Value `def_var`'d into two Variables with the
            // same home, or two disjoint segments of one Variable
            // that failed to merge) — that's fine iff their ranges
            // don't overlap. Reuse the `spillslot_can_fit_spillset`
            // range-btree for exactly that check; on overlap, fall
            // through to auto-allocation (correctness over
            // preference — a redundant `[sp + N]` slot is worse
            // than an aliasing `[fp − r×8]` slot only in
            // instruction count, not soundness). NOT pushed onto
            // `slots_by_class` — auto slots never probe fixed ones.
            if let Some(fixed) = self.ctx.spillsets[spillset].fixed_slot {
                debug_assert!(fixed.is_fixed());
                // `fixed_index() == 0` is the POISON sentinel (a
                // Value shared across two Variables with different
                // fixed homes — see `record_frame_head_binding`).
                // It exists only to make `merge_bundles` refuse a
                // merge with any real fixed slot; here it falls
                // through to auto-allocation.
                if fixed.fixed_index() == 0 {
                    // fall through to auto
                }
                else {
                // One `SpillSlotData` per distinct fixed index,
                // shared across every spillset that fits.
                let idx = match self.ctx.fixed_spillslot_map.get(&fixed) {
                    Some(&idx) => idx,
                    None => {
                        let idx = SpillSlotIndex::new(self.ctx.spillslots.len());
                        let ranges = self
                            .ctx
                            .scratch_spillset_pool
                            .pop()
                            .unwrap_or_default();
                        self.ctx.spillslots.push(SpillSlotData {
                            ranges,
                            alloc: Allocation::stack(fixed),
                            slots: 0,
                        });
                        self.ctx.fixed_spillslot_map.insert(fixed, idx);
                        idx
                    }
                };
                if self.spillslot_can_fit_spillset(idx, spillset) {
                    self.allocate_spillset_to_spillslot(spillset, idx);
                    trace!(
                        " -> spillset{} pinned to fixed slot {:?}",
                        spillset.index(),
                        fixed
                    );
                    n_fixed += 1;
                    continue;
                }
                // Overlapping live range at this fixed home —
                // fall back to an auto slot for THIS spillset.
                // The other occupant keeps the fixed home.
                trace!(
                    " -> spillset{} fixed slot {:?} conflicts; \
                        falling back to auto",
                    spillset.index(),
                    fixed
                );
                } // end fixed_index() != 0
            }
            let class = self.ctx.spillsets[spillset].class as usize;
            // Try a few existing spillslots.
            let mut i = self.ctx.slots_by_class[class].probe_start;
            let mut success = false;
            // Never probe the same element more than once: limit the
            // attempt count to the number of slots in existence.
            for _attempt in
                0..core::cmp::min(self.ctx.slots_by_class[class].slots.len(), MAX_ATTEMPTS)
            {
                // Note: this indexing of `slots` is always valid
                // because either the `slots` list is empty and the
                // iteration limit above consequently means we don't
                // run this loop at all, or else `probe_start` is
                // in-bounds (because it is made so below when we add
                // a slot, and it always takes on the last index `i`
                // after this loop).
                let spillslot = self.ctx.slots_by_class[class].slots[i];

                if self.spillslot_can_fit_spillset(spillslot, spillset) {
                    self.allocate_spillset_to_spillslot(spillset, spillslot);
                    success = true;
                    self.ctx.slots_by_class[class].probe_start = i;
                    break;
                }

                i = self.ctx.slots_by_class[class].next_index(i);
            }

            if !success {
                // Allocate a new spillslot.
                let spillslot = SpillSlotIndex::new(self.ctx.spillslots.len());
                self.ctx.spillslots.push(SpillSlotData {
                    ranges: self.ctx.scratch_spillset_pool.pop().unwrap_or_default(),
                    alloc: Allocation::none(),
                    slots: self.func.spillslot_size(self.ctx.spillsets[spillset].class) as u32,
                });
                self.ctx.slots_by_class[class].slots.push(spillslot);
                self.ctx.slots_by_class[class].probe_start =
                    self.ctx.slots_by_class[class].slots.len() - 1;

                self.allocate_spillset_to_spillslot(spillset, spillslot);
            }
        }

        #[cfg(feature = "std")]
        if std::env::var("AOT_FIXSPILL_TRACE2").is_ok() {
            // For each required spillset, walk EVERY bundle whose
            // spillset points here and dump its ranges' vregs.
            use alloc::vec::Vec;
            let mut ss_vregs: alloc::collections::BTreeMap<usize, Vec<usize>> =
                Default::default();
            for b in 0..self.ctx.bundles.len() {
                let bi = crate::ion::data_structures::LiveBundleIndex::new(b);
                let ss = self.ctx.bundles[bi].spillset;
                if !ss.is_valid() { continue; }
                if !self.ctx.spillsets[ss].required { continue; }
                for e in self.ctx.bundles[bi].ranges.iter() {
                    ss_vregs.entry(ss.index()).or_default()
                        .push(self.ctx.ranges[e.index].vreg.index());
                }
            }
            for (ss, vregs) in ss_vregs.iter_mut() {
                vregs.sort(); vregs.dedup();
                let ssi = SpillSetIndex::new(*ss);
                std::eprintln!(
                    "[fixspill:ra2:req] ss{} fixed={:?} vregs={:?}",
                    ss, self.ctx.spillsets[ssi].fixed_slot, vregs
                );
            }
        }
        #[cfg(feature = "std")]
        if std::env::var("AOT_FIXSPILL_TRACE").is_ok() && n_req > 0 {
            // Also count spillsets that have a fixed_slot but are
            // NOT required (regalloc gave every range a register)
            // — those are the "register-resident win" cases.
            let mut n_fixed_notreq = 0usize;
            let mut n_seed_fixed = 0usize;
            for i in 0..self.ctx.spillsets.len() {
                let ss = SpillSetIndex::new(i);
                if self.ctx.spillsets[ss].fixed_slot.is_some() {
                    n_seed_fixed += 1;
                    if !self.ctx.spillsets[ss].required {
                        n_fixed_notreq += 1;
                    }
                }
            }
            std::eprintln!(
                "[fixspill:ra2] spillsets={} req={} fixed_req={} \
                 fixed_notreq={} auto={} n_seed_fixed={}",
                self.ctx.spillsets.len(),
                n_req,
                n_fixed,
                n_fixed_notreq,
                n_req - n_fixed,
                n_seed_fixed,
            );
        }
        let _ = (n_req, n_fixed);

        // Assign actual slot indices to spillslots. Skip entries
        // whose `alloc` is already set (embedder-fixed slots).
        for i in 0..self.ctx.spillslots.len() {
            if self.ctx.spillslots[i].alloc.is_none() {
                self.ctx.spillslots[i].alloc =
                    self.allocate_spillslot(self.ctx.spillslots[i].slots);
            }
        }

        trace!("spillslot allocator done");
    }

    pub fn allocate_spillslot(&mut self, size: u32) -> Allocation {
        let mut offset = self.ctx.output.num_spillslots as u32;
        // Align up to `size`.
        debug_assert!(size.is_power_of_two());
        offset = (offset + size - 1) & !(size - 1);
        let slot = if self.func.multi_spillslot_named_by_last_slot() {
            offset + size - 1
        } else {
            offset
        };
        offset += size;
        self.ctx.output.num_spillslots = offset as _;
        Allocation::stack(SpillSlot::new(slot as usize))
    }
}
