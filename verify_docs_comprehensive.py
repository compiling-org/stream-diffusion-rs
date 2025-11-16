#!/usr/bin/env python3

import os
import re
import sys
import json

def check_mermaid_diagrams(file_path):
    """Check if Mermaid diagrams are properly formatted in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all Mermaid code blocks
        mermaid_blocks = re.findall(r'```mermaid\s*([\s\S]*?)\s*```', content)
        
        print(f"File: {file_path}")
        print(f"Found {len(mermaid_blocks)} Mermaid diagrams")
        
        # Check each diagram for basic syntax
        diagram_info = []
        for i, diagram in enumerate(mermaid_blocks):
            lines = diagram.strip().split('\n')
            if len(lines) < 2:
                print(f"  Warning: Diagram {i+1} appears to be empty or too short")
                diagram_info.append({
                    'index': i+1,
                    'type': 'unknown',
                    'nodes': 0,
                    'issues': ['Empty or too short']
                })
                continue
                
            diagram_type = lines[0].strip().split()[0] if lines else "unknown"
            print(f"  Diagram {i+1}: {diagram_type}")
            
            # Count nodes for graph diagrams
            nodes = 0
            if diagram_type in ['graph', 'flowchart']:
                nodes = len(re.findall(r'[A-Z][A-Z0-9]*\[', diagram))
                if nodes == 0:
                    print(f"    Warning: No nodes found")
            
            diagram_info.append({
                'index': i+1,
                'type': diagram_type,
                'nodes': nodes,
                'issues': []
            })
            
        print()
        return diagram_info
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def check_roadmap_features(file_path):
    """Check if roadmap features are properly documented"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for roadmap sections
        roadmap_sections = re.findall(r'##*\s*(.*?[Rr]oadmap.*?|.*?[Ff]uture.*?|.*?[Pp]lan.*?)\s*\n([\s\S]*?)(?=\n##|\Z)', content)
        
        print(f"File: {file_path}")
        print(f"Found {len(roadmap_sections)} roadmap/plan sections")
        
        roadmap_info = []
        for section_title, section_content in roadmap_sections:
            print(f"  Section: {section_title.strip()}")
            
            # Look for feature lists
            feature_lists = re.findall(r'[-*]\s*\[.\]\s*(.+)', section_content)
            if feature_lists:
                print(f"    Found {len(feature_lists)} features/plans")
                features = []
                for feature in feature_lists[:10]:  # Show first 10
                    features.append(feature.strip())
                    print(f"      - {feature.strip()}")
                if len(feature_lists) > 10:
                    print(f"      ... and {len(feature_lists) - 10} more")
                
                roadmap_info.append({
                    'section': section_title.strip(),
                    'features': features,
                    'total_features': len(feature_lists)
                })
            else:
                print("    No feature list found")
                roadmap_info.append({
                    'section': section_title.strip(),
                    'features': [],
                    'total_features': 0
                })
        
        print()
        return roadmap_info
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def check_current_state_accuracy(file_path):
    """Check if documentation reflects current state of application"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Checking current state accuracy for: {file_path}")
        
        # Look for indicators of current state
        status_indicators = re.findall(r'##*\s*.*?[Ss]tatus.*?\n([\s\S]*?)(?=\n##|\Z)', content)
        implementation_status = re.findall(r'[Ss]tatus.*?:\s*.*?(?:✅|⚠️|❌|🔄|🚧).*', content)
        version_info = re.findall(r'[Vv]ersion.*?:\s*.*', content)
        last_updated = re.findall(r'[Ll]ast [Uu]pdated.*?:\s*.*', content)
        
        accuracy_info = {
            'status_sections': len(status_indicators),
            'implementation_indicators': len(implementation_status),
            'version_info': len(version_info),
            'last_updated': len(last_updated)
        }
        
        print(f"  Status sections: {accuracy_info['status_sections']}")
        print(f"  Implementation indicators: {accuracy_info['implementation_indicators']}")
        print(f"  Version info entries: {accuracy_info['version_info']}")
        print(f"  Last updated entries: {accuracy_info['last_updated']}")
        print()
        
        return accuracy_info
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def check_flowcharts_and_diagrams(file_path):
    """Check for flowcharts and process diagrams"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for sequence diagrams, flowcharts, and process diagrams
        sequence_diagrams = re.findall(r'```mermaid\s*sequenceDiagram[\s\S]*?```', content)
        flowcharts = re.findall(r'```mermaid\s*graph[\s\S]*?```', content)
        process_diagrams = re.findall(r'```mermaid\s*flowchart[\s\S]*?```', content)
        
        print(f"File: {file_path}")
        print(f"  Sequence diagrams: {len(sequence_diagrams)}")
        print(f"  Flowcharts: {len(flowcharts)}")
        print(f"  Process diagrams: {len(process_diagrams)}")
        print()
        
        return {
            'sequence_diagrams': len(sequence_diagrams),
            'flowcharts': len(flowcharts),
            'process_diagrams': len(process_diagrams)
        }
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def main():
    """Main verification function"""
    print("=== Comprehensive Documentation Verification Script ===\n")
    
    # Define files to check
    doc_files = [
        'README.md',
        'docs/SYNESTHETIC_FRAMEWORK.md',
        'docs/API_DOCS.md',
        'docs/frontend-progress-state.md'
    ]
    
    total_diagrams = 0
    total_roadmaps = 0
    verification_results = {}
    
    # Check each documentation file
    for doc_file in doc_files:
        full_path = os.path.join(os.getcwd(), doc_file)
        if os.path.exists(full_path):
            print(f"--- Checking {doc_file} ---")
            
            # Check Mermaid diagrams
            diagrams = check_mermaid_diagrams(full_path)
            
            # Check roadmap features
            roadmaps = check_roadmap_features(full_path)
            
            # Check current state accuracy
            accuracy = check_current_state_accuracy(full_path)
            
            # Check flowcharts and diagrams
            flowcharts = check_flowcharts_and_diagrams(full_path)
            
            # Store results
            verification_results[doc_file] = {
                'diagrams': len(diagrams),
                'roadmaps': len(roadmaps),
                'accuracy': accuracy,
                'flowcharts': flowcharts
            }
            
            total_diagrams += len(diagrams)
            total_roadmaps += len(roadmaps)
        else:
            print(f"File not found: {full_path}")
            verification_results[doc_file] = {
                'diagrams': 0,
                'roadmaps': 0,
                'accuracy': {},
                'flowcharts': {}
            }
    
    # Summary
    print("=== Verification Summary ===")
    print(f"Total Mermaid diagrams found: {total_diagrams}")
    print(f"Total roadmap sections found: {total_roadmaps}")
    
    # Detailed summary by file
    print("\n=== Detailed Results by File ===")
    for file, results in verification_results.items():
        print(f"\n{file}:")
        print(f"  Diagrams: {results['diagrams']}")
        print(f"  Roadmaps: {results['roadmaps']}")
        if results['flowcharts']:
            print(f"  Sequence diagrams: {results['flowcharts'].get('sequence_diagrams', 0)}")
            print(f"  Flowcharts: {results['flowcharts'].get('flowcharts', 0)}")
            print(f"  Process diagrams: {results['flowcharts'].get('process_diagrams', 0)}")
    
    if total_diagrams == 0:
        print("\n⚠️  WARNING: No Mermaid diagrams found!")
    else:
        print(f"\n✅ Found {total_diagrams} Mermaid diagrams across documentation")
    
    if total_roadmaps == 0:
        print("⚠️  WARNING: No roadmap sections found!")
    else:
        print(f"✅ Found {total_roadmaps} roadmap sections across documentation")
    
    # Save results to JSON file
    with open('doc_verification_results.json', 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print(f"\n📝 Detailed results saved to doc_verification_results.json")

if __name__ == "__main__":
    main()