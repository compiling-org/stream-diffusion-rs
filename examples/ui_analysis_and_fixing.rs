//! Example: UI Analysis and Fixing
//!
//! This example demonstrates how to use the UI analyzer and fixer tools to assess
//! the current state of the web interface and automatically implement missing functionalities.

use stream_diffusion_rs::advanced_ui_analyzer;
use stream_diffusion_rs::ui_fixer;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("🌊 Stream Diffusion RS - UI Analysis and Fixing Example");
    println!("====================================================");
    
    // Get current directory as project root
    let project_root = std::env::current_dir()?.to_string_lossy().to_string();
    println!("Working directory: {}", project_root);
    
    // Step 1: Run advanced UI analysis
    println!("\n📊 Step 1: Running Advanced UI Analysis...");
    let analysis = advanced_ui_analyzer::run_advanced_analysis(&project_root)?;
    
    println!("Analysis completed successfully!");
    println!("Code Coverage: {:.1}%", analysis.summary.code_coverage);
    println!("Fully Functional Components: {}", analysis.summary.fully_functional);
    println!("Incomplete Components: {}", analysis.summary.incomplete);
    
    // Step 2: Generate detailed analysis report
    println!("\n📝 Step 2: Generating Detailed Analysis Report...");
    let analysis_report_path = Path::new("example_ui_analysis_report.txt");
    advanced_ui_analyzer::generate_detailed_report(&project_root, analysis_report_path)?;
    println!("Analysis report saved to: {}", analysis_report_path.display());
    
    // Step 3: Save JSON analysis
    println!("\n💾 Step 3: Saving JSON Analysis...");
    let json_analysis_path = Path::new("example_ui_analysis.json");
    let analyzer = advanced_ui_analyzer::AdvancedUIAnalyzer::new(&project_root);
    analyzer.save_analysis(json_analysis_path)?;
    println!("JSON analysis saved to: {}", json_analysis_path.display());
    
    // Step 4: Display key findings
    println!("\n🔍 Step 4: Key Findings...");
    let incomplete_components: Vec<_> = analysis.components.iter()
        .filter(|c| !c.functional)
        .collect();
    
    if incomplete_components.is_empty() {
        println!("✅ All UI components are fully functional!");
    } else {
        println!("⚠️  Found {} incomplete components:", incomplete_components.len());
        for component in incomplete_components {
            println!("  - {}", component.name);
            if !component.issues.is_empty() {
                println!("    Issues:");
                for issue in &component.issues {
                    println!("      • {}", issue);
                }
            }
            if !component.recommendations.is_empty() {
                println!("    Recommendations:");
                for recommendation in &component.recommendations {
                    println!("      • {}", recommendation);
                }
            }
        }
    }
    
    // Step 5: Run UI fixes
    println!("\n🔧 Step 5: Running UI Fixes...");
    let fix_results = ui_fixer::run_ui_fixes(&project_root)?;
    
    let successful_fixes = fix_results.iter().filter(|r| r.success).count();
    let failed_fixes = fix_results.len() - successful_fixes;
    
    println!("Fix process completed!");
    println!("Successful fixes: {}", successful_fixes);
    println!("Failed fixes: {}", failed_fixes);
    
    // Step 6: Generate fix report
    println!("\n📄 Step 6: Generating Fix Report...");
    let fix_report_path = Path::new("example_ui_fix_report.txt");
    ui_fixer::generate_fix_report(&project_root, &fix_results, fix_report_path)?;
    println!("Fix report saved to: {}", fix_report_path.display());
    
    // Step 7: Display fix results
    println!("\n📋 Step 7: Fix Results Summary...");
    for result in &fix_results {
        let status = if result.success { "✅" } else { "❌" };
        println!("{} {}: {}", status, result.component, result.message);
    }
    
    // Step 8: Run post-fix analysis to show improvements
    println!("\n📈 Step 8: Post-Fix Analysis...");
    let post_fix_analysis = advanced_ui_analyzer::run_advanced_analysis(&project_root)?;
    
    println!("Post-fix code coverage: {:.1}%", post_fix_analysis.summary.code_coverage);
    println!("Improvement: {:+.1}%", post_fix_analysis.summary.code_coverage - analysis.summary.code_coverage);
    
    // Step 9: Final summary
    println!("\n🎉 Step 9: Final Summary...");
    println!("UI Analysis and Fixing Process Completed Successfully!");
    println!("===================================================");
    println!("Initial code coverage: {:.1}%", analysis.summary.code_coverage);
    println!("Final code coverage: {:.1}%", post_fix_analysis.summary.code_coverage);
    println!("Components fixed: {}", successful_fixes);
    println!("Remaining issues: {}", failed_fixes);
    
    if failed_fixes == 0 {
        println!("\n🎊 All UI components have been successfully implemented!");
    } else {
        println!("\n⚠️  Some components still require manual implementation.");
        println!("Please check the fix report for details on failed fixes.");
    }
    
    Ok(())
}