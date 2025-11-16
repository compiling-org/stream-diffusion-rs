#!/usr/bin/env python3

import os
import re
import json

def analyze_source_code():
    """Analyze the source code structure to understand the actual application architecture"""
    print("=== Analyzing Source Code Structure ===\n")
    
    src_path = "src"
    if not os.path.exists(src_path):
        print("Source directory not found")
        return {}
    
    # Find all Rust files
    rust_files = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.rs'):
                rust_files.append(os.path.join(root, file))
    
    print(f"Found {len(rust_files)} Rust source files:")
    for file in rust_files:
        print(f"  - {file}")
    
    # Analyze key components
    components = {
        'modules': [],
        'structs': [],
        'traits': [],
        'functions': []
    }
    
    # Look for key components in the code
    for file_path in rust_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract module names
            module_matches = re.findall(r'pub\s+mod\s+(\w+)', content)
            for module in module_matches:
                if module not in components['modules']:
                    components['modules'].append(module)
            
            # Extract struct names
            struct_matches = re.findall(r'pub\s+struct\s+(\w+)', content)
            for struct in struct_matches:
                if struct not in components['structs']:
                    components['structs'].append(struct)
            
            # Extract trait names
            trait_matches = re.findall(r'pub\s+trait\s+(\w+)', content)
            for trait in trait_matches:
                if trait not in components['traits']:
                    components['traits'].append(trait)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"\nIdentified components:")
    print(f"  Modules: {len(components['modules'])}")
    for module in sorted(components['modules']):
        print(f"    - {module}")
    
    print(f"  Structs: {len(components['structs'])}")
    for struct in sorted(components['structs'])[:10]:  # Show first 10
        print(f"    - {struct}")
    if len(components['structs']) > 10:
        print(f"    ... and {len(components['structs']) - 10} more")
    
    print(f"  Traits: {len(components['traits'])}")
    for trait in sorted(components['traits']):
        print(f"    - {trait}")
    
    print()
    return components

def check_documentation_alignment(components):
    """Check if documentation aligns with actual code structure"""
    print("=== Checking Documentation Alignment ===\n")
    
    doc_files = [
        'README.md',
        'docs/SYNESTHETIC_FRAMEWORK.md',
        'docs/API_DOCS.md',
        'docs/frontend-progress-state.md'
    ]
    
    alignment_results = {}
    
    for doc_file in doc_files:
        full_path = os.path.join(os.getcwd(), doc_file)
        if not os.path.exists(full_path):
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for mentions of actual components
            found_modules = []
            found_structs = []
            found_traits = []
            
            for module in components['modules']:
                if module.lower() in content.lower():
                    found_modules.append(module)
            
            for struct in components['structs']:
                if struct.lower() in content.lower():
                    found_structs.append(struct)
            
            for trait in components['traits']:
                if trait.lower() in content.lower():
                    found_traits.append(trait)
            
            alignment_results[doc_file] = {
                'modules_mentioned': len(found_modules),
                'structs_mentioned': len(found_structs),
                'traits_mentioned': len(found_traits),
                'total_components_mentioned': len(found_modules) + len(found_structs) + len(found_traits)
            }
            
            print(f"{doc_file}:")
            print(f"  Modules mentioned: {len(found_modules)}/{len(components['modules'])}")
            print(f"  Structs mentioned: {len(found_structs)}/{len(components['structs'])}")
            print(f"  Traits mentioned: {len(found_traits)}/{len(components['traits'])}")
            print()
            
        except Exception as e:
            print(f"Error reading {doc_file}: {e}")
            alignment_results[doc_file] = {
                'modules_mentioned': 0,
                'structs_mentioned': 0,
                'traits_mentioned': 0,
                'total_components_mentioned': 0
            }
    
    return alignment_results

def verify_roadmap_feasibility(components):
    """Verify if roadmap features are feasible based on current code structure"""
    print("=== Verifying Roadmap Feasibility ===\n")
    
    # Look for roadmap in frontend-progress-state.md
    roadmap_file = 'docs/frontend-progress-state.md'
    full_path = os.path.join(os.getcwd(), roadmap_file)
    
    if not os.path.exists(full_path):
        print("Roadmap file not found")
        return {}
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract roadmap features
        roadmap_sections = re.findall(r'##*\s*(.*?[Rr]oadmap.*?|.*?[Ff]uture.*?|.*?[Pp]lan.*?)\s*\n([\s\S]*?)(?=\n##|\Z)', content)
        
        feasibility_results = {}
        
        for section_title, section_content in roadmap_sections:
            print(f"Checking feasibility for: {section_title.strip()}")
            
            feature_lists = re.findall(r'[-*]\s*\[.\]\s*(.+)', section_content)
            
            feasible_count = 0
            total_features = len(feature_lists)
            
            # Simple feasibility check based on existing components
            for feature in feature_lists:
                # Check if feature mentions existing components
                has_existing_component = False
                for component_list in [components['modules'], components['structs'], components['traits']]:
                    for component in component_list:
                        if component.lower() in feature.lower():
                            has_existing_component = True
                            break
                    if has_existing_component:
                        break
                
                if has_existing_component:
                    feasible_count += 1
            
            feasibility_results[section_title.strip()] = {
                'total_features': total_features,
                'feasible_features': feasible_count,
                'feasibility_percentage': (feasible_count / total_features * 100) if total_features > 0 else 0
            }
            
            print(f"  Total features: {total_features}")
            print(f"  Feasible features: {feasible_count}")
            print(f"  Feasibility: {feasibility_results[section_title.strip()]['feasibility_percentage']:.1f}%")
            print()
        
        return feasibility_results
        
    except Exception as e:
        print(f"Error reading roadmap: {e}")
        return {}

def main():
    """Main verification function"""
    print("=== Code-Documentation Alignment Verification ===\n")
    
    # Analyze source code
    components = analyze_source_code()
    
    # Check documentation alignment
    alignment = check_documentation_alignment(components)
    
    # Verify roadmap feasibility
    feasibility = verify_roadmap_feasibility(components)
    
    # Summary
    print("=== Final Summary ===")
    print(f"Total components identified in code: {len(components['modules']) + len(components['structs']) + len(components['traits'])}")
    print(f"  Modules: {len(components['modules'])}")
    print(f"  Structs: {len(components['structs'])}")
    print(f"  Traits: {len(components['traits'])}")
    
    total_mentioned = sum(result['total_components_mentioned'] for result in alignment.values())
    print(f"\nTotal component references in documentation: {total_mentioned}")
    
    if feasibility:
        avg_feasibility = sum(result['feasibility_percentage'] for result in feasibility.values()) / len(feasibility)
        print(f"Average roadmap feasibility: {avg_feasibility:.1f}%")
    
    # Save results
    results = {
        'components': components,
        'alignment': alignment,
        'feasibility': feasibility
    }
    
    with open('code_alignment_verification.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📝 Detailed results saved to code_alignment_verification.json")

if __name__ == "__main__":
    main()