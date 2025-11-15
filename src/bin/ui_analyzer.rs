//! Command-line interface for the UI Analyzer

use stream_diffusion_rs::ui_analyzer;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Stream Diffusion RS - UI Analyzer");
    println!("====================================");
    
    // Run analysis
    match ui_analyzer::run_analysis() {
        Ok(analysis) => {
            println!("\n📊 UI Analysis Results:");
            println!("======================");
            println!("Overall Status: {}", analysis.overall_status);
            
            println!("\n✅ Fully Implemented Features:");
            if analysis.fully_implemented.is_empty() {
                println!("  None");
            } else {
                for feature in &analysis.fully_implemented {
                    println!("  - {}", feature);
                }
            }
            
            println!("\n🟡 Partially Implemented Features:");
            if analysis.partially_implemented.is_empty() {
                println!("  None");
            } else {
                for feature in &analysis.partially_implemented {
                    println!("  - {}", feature);
                }
            }
            
            println!("\n❌ Missing Features:");
            if analysis.missing_features.is_empty() {
                println!("  None");
            } else {
                for feature in &analysis.missing_features {
                    println!("  - {}", feature);
                }
            }
            
            println!("\n💡 Implementation Recommendations:");
            for recommendation in &analysis.recommendations {
                println!("  - {}", recommendation);
            }
            
            // Save detailed report
            let report_path = Path::new("ui_analysis_report.txt");
            match ui_analyzer::generate_detailed_report(report_path) {
                Ok(_) => println!("\n📝 Detailed report saved to: {}", report_path.display()),
                Err(e) => eprintln!("⚠️  Failed to save report: {}", e),
            }
            
            // Save JSON analysis
            let json_path = Path::new("ui_analysis.json");
            let analyzer = ui_analyzer::UIAnalyzer::new();
            match analyzer.save_analysis(json_path) {
                Ok(_) => println!("💾 JSON analysis saved to: {}", json_path.display()),
                Err(e) => eprintln!("⚠️  Failed to save JSON analysis: {}", e),
            }
        }
        Err(e) => {
            eprintln!("❌ Analysis failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}