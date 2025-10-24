//! Data visualization utilities for ML research and EEG analysis

use plotters::prelude::*;
use plotters::style::colors::*;
use ndarray::Array2;
use std::path::Path;

/// Generic plotting utilities
pub struct Plotter {
    output_dir: std::path::PathBuf,
}

impl Plotter {
    pub fn new(output_dir: &Path) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
        }
    }

    /// Plot line series
    pub fn plot_line(&self, x_data: &[f32], y_data: &[f32], title: &str, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.output_dir.join(filename);

        let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let x_min = x_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let x_max = x_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let y_max = y_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?;

        root.present()?;
        Ok(())
    }

    /// Plot scatter plot
    pub fn plot_scatter(&self, x_data: &[f32], y_data: &[f32], title: &str, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.output_dir.join(filename);

        let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let x_min = x_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let x_max = x_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let y_max = y_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(PointSeries::of_element(
            x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
            2,
            &BLUE,
            &|c, s, st| {
                return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
            },
        ))?;

        root.present()?;
        Ok(())
    }

    /// Plot histogram
    pub fn plot_histogram(&self, data: &[f32], bins: usize, title: &str, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.output_dir.join(filename);

        let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_val..max_val, 0..100)?;

        chart.configure_mesh().draw()?;

        let hist = self.compute_histogram(data, bins, min_val, max_val);

        chart.draw_series(hist.iter().enumerate().map(|(i, &count)| {
            let bin_start = min_val + (max_val - min_val) * i as f32 / bins as f32;
            let bin_end = min_val + (max_val - min_val) * (i + 1) as f32 / bins as f32;
            Rectangle::new([(bin_start, 0), (bin_end, count)], BLUE.filled())
        }))?;

        root.present()?;
        Ok(())
    }

    /// Plot confusion matrix
    pub fn plot_confusion_matrix(&self, matrix: &Array2<f32>, class_names: &[&str], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.output_dir.join(filename);

        let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Confusion Matrix", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..matrix.ncols(), 0..matrix.nrows())?;

        chart.configure_mesh()
            .x_labels(matrix.ncols())
            .y_labels(matrix.nrows())
            .x_label_formatter(&|x| if *x < class_names.len() { class_names[*x].to_string() } else { "".to_string() })
            .y_label_formatter(&|y| if *y < class_names.len() { class_names[*y].to_string() } else { "".to_string() })
            .draw()?;

        // Plot heatmap
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let value = matrix[[i, j]];
                let color = self.value_to_color(value);

                chart.draw_series(std::iter::once(Rectangle::new(
                    [(j, i), (j + 1, i + 1)],
                    color.filled(),
                )))?;

                // Add text if value is significant
                if value > 0.1 {
                    chart.draw_series(std::iter::once(Text::new(
                        format!("{:.2}", value),
                        (j, i),
                        ("sans-serif", 12).into_font().color(&BLACK),
                    )))?;
                }
            }
        }

        root.present()?;
        Ok(())
    }

    /// Plot training curves
    pub fn plot_training_curves(&self, train_losses: &[f32], val_losses: &[f32], train_accs: Option<&[f32]>, val_accs: Option<&[f32]>, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.output_dir.join(filename);

        let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let epochs: Vec<usize> = (0..train_losses.len()).collect();
        let epochs_f32: Vec<f32> = epochs.iter().map(|&x| x as f32).collect();

        let loss_min = train_losses.iter().chain(val_losses.iter()).fold(f32::INFINITY, |a, &b| a.min(b));
        let loss_max = train_losses.iter().chain(val_losses.iter()).fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption("Training Curves", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..epochs_f32.last().unwrap_or(&1.0), loss_min..loss_max)?;

        chart.configure_mesh().draw()?;

        // Plot losses
        chart.draw_series(LineSeries::new(
            epochs_f32.iter().zip(train_losses.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?.label("Train Loss").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.draw_series(LineSeries::new(
            epochs_f32.iter().zip(val_losses.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?.label("Val Loss").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Plot accuracies if provided
        if let (Some(train_accs), Some(val_accs)) = (train_accs, val_accs) {
            let acc_min = train_accs.iter().chain(val_accs.iter()).fold(f32::INFINITY, |a, &b| a.min(b));
            let acc_max = train_accs.iter().chain(val_accs.iter()).fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut chart_acc = ChartBuilder::on(&root)
                .caption("Training Curves", ("sans-serif", 20))
                .margin(5)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(0.0..epochs_f32.last().unwrap_or(&1.0), acc_min..acc_max)?;

            chart_acc.configure_mesh().draw()?;

            chart_acc.draw_series(LineSeries::new(
                epochs_f32.iter().zip(train_accs.iter()).map(|(&x, &y)| (x, y)),
                &GREEN,
            ))?.label("Train Acc").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

            chart_acc.draw_series(LineSeries::new(
                epochs_f32.iter().zip(val_accs.iter()).map(|(&x, &y)| (x, y)),
                &RGBColor(255, 165, 0),
            ))?.label("Val Acc").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        }

        root.present()?;
        Ok(())
    }

    /// Plot EEG topography
    pub fn plot_eeg_topography(&self, values: &[f32], electrode_names: &[&str], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.output_dir.join(filename);

        let root = BitMapBackend::new(&output_path, (600, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("EEG Topography", ("sans-serif", 20))
            .build_cartesian_2d(-1.0..1.0, -1.0..1.0)?;

        chart.configure_mesh().draw()?;

        // Standard 10-20 electrode positions (simplified)
        let positions = vec![
            (0.0, 0.8, "Fz"), (0.6, 0.6, "F4"), (-0.6, 0.6, "F3"),
            (0.8, 0.0, "C4"), (-0.8, 0.0, "C3"), (0.6, -0.6, "P4"),
            (-0.6, -0.6, "P3"), (0.0, -0.8, "Pz"), (0.0, 0.0, "Cz"),
        ];

        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        for (i, &(x, y, name)) in positions.iter().enumerate() {
            if i < values.len() {
                let value = values[i];
                let normalized_value = (value - min_val) / (max_val - min_val);
                let color = self.interpolate_color(normalized_value);

                chart.draw_series(std::iter::once(Circle::new((x, y), 8, color.filled())))?;

                chart.draw_series(std::iter::once(Text::new(
                    name,
                    (x, y + 0.05),
                    ("sans-serif", 10).into_font().color(&BLACK),
                )))?;
            }
        }

        // Add colorbar
        self.draw_colorbar(&root, min_val, max_val)?;

        root.present()?;
        Ok(())
    }

    /// Plot spectrogram
    pub fn plot_spectrogram(&self, spectrogram: &Array2<f32>, freqs: &[f32], times: &[f32], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.output_dir.join(filename);

        let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Spectrogram", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(times[0]..times.last().unwrap_or(&1.0), freqs[0]..freqs.last().unwrap_or(&100.0))?;

        chart.configure_mesh().draw()?;

        let min_val = spectrogram.fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = spectrogram.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        for i in 0..spectrogram.nrows() {
            for j in 0..spectrogram.ncols() {
                let value = spectrogram[[i, j]];
                let normalized_value = (value - min_val) / (max_val - min_val);
                let color = self.interpolate_color(normalized_value);

                chart.draw_series(std::iter::once(Rectangle::new(
                    [(times[j], freqs[i]), (times[j + 1], freqs[i + 1])],
                    color.filled(),
                )))?;
            }
        }

        root.present()?;
        Ok(())
    }

    // Helper methods
    fn compute_histogram(&self, data: &[f32], bins: usize, min_val: f32, max_val: f32) -> Vec<i32> {
        let mut hist = vec![0i32; bins];
        let bin_width = (max_val - min_val) / bins as f32;

        for &value in data {
            let bin = ((value - min_val) / bin_width) as usize;
            let bin = bin.min(bins - 1);
            hist[bin] += 1;
        }

        hist
    }

    fn value_to_color(&self, value: f32) -> RGBColor {
        let normalized = value.max(0.0).min(1.0);
        let r = (normalized * 255.0) as u8;
        let g = ((1.0 - normalized) * 255.0) as u8;
        let b = 128;
        RGBColor(r, g, b)
    }

    fn interpolate_color(&self, t: f32) -> RGBColor {
        // Blue to red colormap
        let r = (t * 255.0) as u8;
        let g = ((1.0 - t) * 255.0) as u8;
        let b = 128;
        RGBColor(r, g, b)
    }

    fn draw_colorbar(&self, root: &DrawingArea<BitMapBackend, plotters::coord::Shift>, min_val: f32, max_val: f32) -> Result<(), Box<dyn std::error::Error>> {
        let colorbar_area = root.split_evenly((1, 2))[1];

        let mut colorbar = ChartBuilder::on(&colorbar_area)
            .margin(20)
            .build_cartesian_2d(0..1, min_val..max_val)?;

        colorbar.configure_mesh()
            .set_tick_mark_size((0, 0))
            .draw()?;

        // Draw colorbar gradient
        for i in 0..100 {
            let t = i as f32 / 99.0;
            let color = self.interpolate_color(t);
            colorbar.draw_series(std::iter::once(Rectangle::new(
                [(0, (min_val + t * (max_val - min_val)) as i32), (1, (min_val + (t + 0.01) * (max_val - min_val)) as i32)],
                color.filled(),
            )))?;
        }

        Ok(())
    }
}

/// Real-time visualization for streaming data
pub struct RealTimeVisualizer {
    plotter: Plotter,
    buffer_size: usize,
    data_buffer: Vec<f32>,
}

impl RealTimeVisualizer {
    pub fn new(output_dir: &Path, buffer_size: usize) -> Self {
        Self {
            plotter: Plotter::new(output_dir),
            buffer_size,
            data_buffer: Vec::new(),
        }
    }

    /// Add data point to buffer
    pub fn add_data_point(&mut self, value: f32) {
        self.data_buffer.push(value);
        if self.data_buffer.len() > self.buffer_size {
            self.data_buffer.remove(0);
        }
    }

    /// Update real-time plot
    pub fn update_plot(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let x_data: Vec<f32> = (0..self.data_buffer.len()).map(|i| i as f32).collect();
        self.plotter.plot_line(&x_data, &self.data_buffer, "Real-time Data", filename)
    }

    /// Get current buffer
    pub fn get_buffer(&self) -> &[f32] {
        &self.data_buffer
    }
}

/// Dashboard for ML experiment monitoring
pub struct ExperimentDashboard {
    plotter: Plotter,
    metrics_history: std::collections::HashMap<String, Vec<f32>>,
}

impl ExperimentDashboard {
    pub fn new(output_dir: &Path) -> Self {
        Self {
            plotter: Plotter::new(output_dir),
            metrics_history: std::collections::HashMap::new(),
        }
    }

    /// Update metric
    pub fn update_metric(&mut self, name: &str, value: f32) {
        self.metrics_history.entry(name.to_string()).or_insert(Vec::new()).push(value);
    }

    /// Generate dashboard plots
    pub fn generate_dashboard(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Plot all metrics
        for (name, values) in &self.metrics_history {
            let x_data: Vec<f32> = (0..values.len()).map(|i| i as f32).collect();
            self.plotter.plot_line(&x_data, values, &format!("{} over time", name), &format!("{}_history.png", name))?;
        }

        // Generate summary statistics
        let mut summary = String::new();
        summary.push_str("Experiment Summary\n");
        summary.push_str("==================\n\n");

        for (name, values) in &self.metrics_history {
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            summary.push_str(&format!("{}:\n", name));
            summary.push_str(&format!("  Mean: {:.4}\n", mean));
            summary.push_str(&format!("  Min: {:.4}\n", min));
            summary.push_str(&format!("  Max: {:.4}\n", max));
            summary.push_str(&format!("  Final: {:.4}\n\n", values.last().unwrap_or(&0.0)));
        }

        std::fs::write(self.plotter.output_dir.join("experiment_summary.txt"), summary)?;

        Ok(())
    }
}