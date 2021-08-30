use crate::{ClipSpheres, GenerateCommands, MapConfig};

use building_blocks::storage::prelude::*;

use bevy_utilities::bevy::{prelude::*, utils::tracing};

pub fn detect_new_chunks_system(
    config: Res<MapConfig>,
    clip_spheres: Res<ClipSpheres>,
    gen_commands: Res<GenerateCommands>,
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
            |new_slot| new_slots.push(new_slot),
        );
        gen_commands.add_new_slots(new_slots.into_iter());
    }
}
