            for channel in 0..buffer.channels {
                let mut delayed_samples = Vec::new();
                {
                    let channel_data = buffer.channel(channel);
                    for i in 0..channel_data.len() {
                        // Modulate delay time with LFO
                        let lfo = (self.phase + i as f32 / buffer.samples_per_channel() as f32 * 2.0 * PI).sin();
                        let mod_delay = delay_samples + (lfo * depth_samples as f32) as usize;
                        
                        // Read delayed sample (simple linear interpolation)
                        let delay_idx = i.saturating_sub(mod_delay);
                        if delay_idx < channel_data.len() {
                            delayed_samples.push(channel_data[delay_idx]);
                        } else {
                            delayed_samples.push(0.0);
                        }
                    }
                }
                
                // Apply the delayed samples
                let channel_data = buffer.channel_mut(channel);
                for (i, sample) in channel_data.iter_mut().enumerate() {
                    *sample = *sample * (1.0 - self.mix) + delayed_samples[i] * self.mix;
                }
            }