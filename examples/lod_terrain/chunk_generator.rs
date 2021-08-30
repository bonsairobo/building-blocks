use crate::voxel_map::VoxelMap;

use building_blocks::prelude::ClipmapSlot3;

use bevy_utilities::bevy::{prelude::*, utils::tracing};

use std::sync::Mutex;

/// Inserts new chunks into the `ChunkTree` after they are generated.
///
/// This is a separate system from chunk_generator_system so chunk generation can run in parallel with other systems that borrow
/// the `ChunkTree`.
pub fn new_chunk_filler_system<Map: VoxelMap>(
    commands: Res<GenerateCommands>,
    mut map: ResMut<Map>,
) {
    let new_slots: Vec<_> = commands.new_slots.lock().unwrap().drain(..).collect();
    for slot in new_slots.into_iter() {
        map.mark_node_for_loading_if_vacant(slot.key);
    }
}

/// Manages new chunk generation by searching the chunk tree for nodes that need to be loaded.
///
/// Generated chunks will be sent to the new_chunk_filler_system, since it can mutate the chunk tree.
pub fn chunk_generator_system<Map: VoxelMap>() {
    let generator_span = tracing::info_span!("chunk_generator");
    let _trace_guard = generator_span.enter();
}

#[derive(Default)]
pub struct GenerateCommands {
    /// New chunks that entered the clip sphere this frame.
    new_slots: Mutex<Vec<ClipmapSlot3>>,
}

impl GenerateCommands {
    pub fn add_new_slots(&self, new_slots: impl Iterator<Item = ClipmapSlot3>) {
        self.new_slots.lock().unwrap().extend(new_slots);
    }
}
