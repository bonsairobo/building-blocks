use std::convert::TryInto;
use std::time::Duration;

pub struct FrameBudget {
    num_threads: u32,
    item_time_estimate_us: u32,
    target_frame_time_us: u32,
    timer: Option<WorkTimer>,
}

struct WorkTimer {
    total_cpu_time: Duration,
    items_completed: u32,
}

impl WorkTimer {
    pub fn start() -> Self {
        Self {
            total_cpu_time: Duration::new(0, 0),
            items_completed: 0,
        }
    }

    pub fn complete_item(&mut self, d: Duration) {
        self.total_cpu_time += d;
        self.items_completed += 1;
    }

    pub fn average_cpu_time_us(&self) -> u32 {
        let frame_cpu_time_us: u32 = self
            .total_cpu_time
            .as_micros()
            .try_into()
            .unwrap_or(u32::MAX);

        frame_cpu_time_us / self.items_completed.max(1)
    }
}

impl FrameBudget {
    pub fn new(
        num_threads: u32,
        target_frame_time_us: u32,
        initial_item_time_estimate_us: u32,
    ) -> Self {
        Self {
            num_threads,
            target_frame_time_us,
            item_time_estimate_us: initial_item_time_estimate_us,
            timer: None,
        }
    }

    pub fn request_work(&self, items_in_queue: u32) -> u32 {
        self.items_per_frame() - items_in_queue
    }

    pub fn reset_timer(&mut self) {
        self.timer = Some(WorkTimer::start());
    }

    pub fn complete_item(&mut self, cpu_time: Duration) {
        let timer = self.timer.as_mut().expect("Start the timer first");
        timer.complete_item(cpu_time);
    }

    pub fn update_estimate(&mut self) {
        if let Some(timer) = self.timer.as_ref() {
            if timer.items_completed > 0 {
                self.item_time_estimate_us = timer.average_cpu_time_us();
            }
        }
    }

    fn items_per_frame(&self) -> u32 {
        (self.target_frame_time_us * self.num_threads) / self.item_time_estimate_us.max(1)
    }
}
