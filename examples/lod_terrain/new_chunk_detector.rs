use crate::{ClipSpheres, MapConfig};

use building_blocks::storage::prelude::*;

use bevy_utilities::bevy::{prelude::*, utils::tracing};

pub fn detect_new_chunks_system(config: Res<MapConfig>, clip_spheres: Res<ClipSpheres>) {
    let indexer = ChunkIndexer3::new(config.chunk_shape());

    let clip_events_span = tracing::info_span!("clip_events");
    {
        let _trace_guard = clip_events_span.enter();
        clipmap_new_chunks(
            &indexer,
            config.root_lod(),
            config.detect_enter_lod,
            config.detail,
            clip_spheres.old_sphere,
            clip_spheres.new_sphere,
            |_| (),
        );
    }
}
