//! Advanced Command-line interface for the UI Analyzer

use stream_diffusion_rs::advanced_ui_analyzer;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Stream Diffusion RS - Advanced UI Analyzer");
    println!("============================================");
    
    // Get current directory as project root
    let project_root = std::env::current_dir()?.to_string_lossy().to_string();
    println!("Analyzing project at: {}", project_root);
    
    // Run analysis
    match advanced_ui_analyzer::run_advanced_analysis(&project_root) {
        Ok(analysis) => {
            println!("\n📊 Advanced UI Analysis Results:");
            println!("===============================");
            println!("Analysis Timestamp: {}", analysis.timestamp);
            println!("Code Coverage: {:.1}%", analysis.summary.code_coverage);
            
            println!("\n📈 Summary:");
            println!("  Total Components: {}", analysis.summary.total_components);
            println!("  Fully Functional: {}", analysis.summary.fully_functional);
            println!("  Backend Only: {}", analysis.summary.backend_only);
            println!("  Frontend Only: {}", analysis.summary.frontend_only);
            println!("  Incomplete: {}", analysis.summary.incomplete);
            
            println!("\n🔍 Component Details:");
            for component in &analysis.components {
                println!("\n{}", component.name);
                println!("  Status: {}", if component.functional { "✅ Functional" } else { "❌ Incomplete" });
                println!("  Frontend: {}", if component.frontend_exists { "✅ Exists" } else { "❌ Missing" });
                println!("  Backend: {}", if component.backend_implemented { "✅ Implemented" } else { "❌ Missing" });
                
                if !component.issues.is_empty() {
                    println!("  Issues:");
                    for issue in &component.issues {
                        println!("    - {}", issue);
                    }
                }
                
                if !component.recommendations.is_empty() {
                    println!("  Recommendations:");
                    for recommendation in &component.recommendations {
                        println!("    - {}", recommendation);
                    }
                }
            }
            
            // Save detailed report
            let report_path = Path::new("advanced_ui_analysis_report.txt");
            match advanced_ui_analyzer::generate_detailed_report(&project_root, report_path) {
                Ok(_) => println!("\n📝 Detailed report saved to: {}", report_path.display()),
                Err(e) => eprintln!("⚠️  Failed to save report: {}", e),
            }
            
            // Save JSON analysis
            let json_path = Path::new("advanced_ui_analysis.json");
            let analyzer = advanced_ui_analyzer::AdvancedUIAnalyzer::new(&project_root);
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