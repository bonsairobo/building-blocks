use std::{convert::TryInto, time::Duration};

pub struct FrameBudget {
    item_time_estimate_us: u32,
    target_frame_time_us: u32,
}

impl FrameBudget {
    pub fn new(target_frame_time_us: u32, initial_item_time_estimate_us: u32) -> Self {
        Self {
            target_frame_time_us,
            item_time_estimate_us: initial_item_time_estimate_us,
        }
    }

    pub fn request_work(&self, items_in_queue: u32) -> u32 {
        self.items_per_frame() - items_in_queue
    }

    pub fn update_estimate(&mut self, frame_time_elapsed: Duration) {
        let frame_time_us: u32 = frame_time_elapsed
            .as_micros()
            .try_into()
            .unwrap_or(u32::MAX);
        self.item_time_estimate_us = frame_time_us / self.items_per_frame().max(1);
    }

    fn items_per_frame(&self) -> u32 {
        self.target_frame_time_us / self.item_time_estimate_us.max(1)
    }
}
