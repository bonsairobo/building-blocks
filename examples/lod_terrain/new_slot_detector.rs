use crate::{chunk_generator::NewSlot, ClipSpheres, MapConfig, SyncBatch};

use building_blocks::storage::prelude::*;

use bevy_utilities::bevy::{prelude::*, utils::tracing};

pub fn detect_new_slots_system(
    config: Res<MapConfig>,
    clip_spheres: Res<ClipSpheres>,
    frame_new_slots: Res<SyncBatch<NewSlot>>,
) {
    let indexer = ChunkIndexer3::new(config.chunk_shape());

    let new_chunks_span = tracing::info_span!("clip_events");
    {
        let _trace_guard = new_chunks_span.enter();
        let mut new_slots = Vec::new();
        clipmap_new_chunks(
            &indexer,
            config.root_lod(),
            config.detect_enter_lod,
            config.detail,
            clip_spheres.old_sphere,
            clip_spheres.new_sphere,
            |new_slot| {
                if new_slot.key.minimum.y() < 0 && new_slot.key.minimum.y() >= -64 {
                    new_slots.push(new_slot)
                }
            },
        );
        frame_new_slots.extend(new_slots.into_iter().map(|s| NewSlot { key: s.key }));
    }
}
