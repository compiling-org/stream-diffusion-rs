#!/usr/bin/env python3

import os
import re
import sys

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
        for i, diagram in enumerate(mermaid_blocks):
            lines = diagram.strip().split('\n')
            if len(lines) < 2:
                print(f"  Warning: Diagram {i+1} appears to be empty or too short")
                continue
                
            diagram_type = lines[0].strip().split()[0] if lines else "unknown"
            print(f"  Diagram {i+1}: {diagram_type}")
            
            # Check for common syntax issues
            if diagram_type in ['graph', 'flowchart']:
                nodes = re.findall(r'([A-Z][A-Z0-9]*)\[', diagram)
                if len(nodes) == 0:
                    print(f"    Warning: No nodes found")
            
        print()
        return len(mermaid_blocks)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def check_roadmap_features(file_path):
    """Check if roadmap features are properly documented"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for roadmap sections
        roadmap_sections = re.findall(r'##*\s*(.*?[Rr]oadmap.*?|.*?[Ff]uture.*?|.*?[Pp]lan.*?)\s*\n([\s\S]*?)(?=\n##|\Z)', content)
        
        print(f"File: {file_path}")
        print(f"Found {len(roadmap_sections)} roadmap/plan sections")
        
        for section_title, section_content in roadmap_sections:
            print(f"  Section: {section_title.strip()}")
            
            # Look for feature lists
            feature_lists = re.findall(r'[-*]\s*\[.\]\s*(.+)', section_content)
            if feature_lists:
                print(f"    Found {len(feature_lists)} features/plans")
                for feature in feature_lists[:5]:  # Show first 5
                    print(f"      - {feature.strip()}")
                if len(feature_lists) > 5:
                    print(f"      ... and {len(feature_lists) - 5} more")
            else:
                print("    No feature list found")
        
        print()
        return len(roadmap_sections)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def main():
    """Main verification function"""
    print("=== Documentation Verification Script ===\n")
    
    # Define files to check
    doc_files = [
        'README.md',
        'docs/SYNESTHETIC_FRAMEWORK.md',
        'docs/API_DOCS.md',
        'docs/frontend-progress-state.md'
    ]
    
    total_diagrams = 0
    total_roadmaps = 0
    
    # Check each documentation file
    for doc_file in doc_files:
        full_path = os.path.join(os.getcwd(), doc_file)
        if os.path.exists(full_path):
            print(f"--- Checking {doc_file} ---")
            diagrams = check_mermaid_diagrams(full_path)
            roadmaps = check_roadmap_features(full_path)
            total_diagrams += diagrams
            total_roadmaps += roadmaps
        else:
            print(f"File not found: {full_path}")
    
    # Summary
    print("=== Verification Summary ===")
    print(f"Total Mermaid diagrams found: {total_diagrams}")
    print(f"Total roadmap sections found: {total_roadmaps}")
    
    if total_diagrams == 0:
        print("\n⚠️  WARNING: No Mermaid diagrams found!")
    else:
        print(f"\n✅ Found {total_diagrams} Mermaid diagrams across documentation")
    
    if total_roadmaps == 0:
        print("⚠️  WARNING: No roadmap sections found!")
    else:
        print(f"✅ Found {total_roadmaps} roadmap sections across documentation")

if __name__ == "__main__":
    main()