use crate::{chunk_generator::NewSlot, ClipSpheres, MapConfig, SyncBatch};

use building_blocks::storage::prelude::*;

use bevy_utilities::bevy::{prelude::*, utils::tracing};

pub fn detect_new_slots_system(
    config: Res<MapConfig>,
    clip_spheres: Res<ClipSpheres>,
    frame_new_slots: Res<SyncBatch<NewSlot>>,
) {
    let span = tracing::info_span!("clip_events");
    let _trace_guard = span.enter();

    let indexer = ChunkIndexer3::new(config.chunk_shape());
    let mut new_slots = Vec::new();
    clipmap_new_chunks_intersecting_sphere(
        &indexer,
        config.root_lod(),
        config.detect_enter_lod,
        config.detail,
        clip_spheres.old_sphere,
        clip_spheres.new_sphere,
        |new_slot| new_slots.push(new_slot),
    );
    frame_new_slots.extend(new_slots.into_iter().map(|s| NewSlot { key: s.key }));
}
