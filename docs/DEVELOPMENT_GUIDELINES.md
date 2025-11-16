# Development Guidelines and Disciplinary Script

## Core Principles

1. **Never corrupt files**: Always ensure file integrity is maintained
2. **Make surgical edits**: Only modify the specific lines that need changes
3. **Verify before committing**: Always check that changes work before committing
4. **Backup before major changes**: Create checkpoints before significant modifications

## File Handling Rules

1. **Always read the full context** before making changes
2. **Never replace entire files** unless explicitly instructed
3. **Preserve existing structure** and only modify specific functions/sections
4. **Use proper error handling** when files are corrupted

## Error Recovery Protocol

1. If a file becomes corrupted:
   - Immediately restore from git: `git checkout HEAD -- <file>`
   - Identify the specific issue that caused corruption
   - Make targeted fixes rather than broad changes
   - Verify the fix works before proceeding

## Code Change Process

1. **Read**: Understand the existing code structure
2. **Plan**: Identify exactly what needs to be changed
3. **Edit**: Make minimal, surgical changes
4. **Verify**: Check that changes compile and work
5. **Commit**: Only commit working changes

## Common Mistakes to Avoid

1. Putting code outside of functions/classes
2. Removing imports or struct definitions accidentally
3. Making changes to the wrong part of a file
4. Not handling borrowing issues properly in Rust

## When Things Go Wrong

1. Stop making changes immediately
2. Restore corrupted files from git
3. Reassess the approach
4. Make smaller, more targeted changes
5. Test frequently