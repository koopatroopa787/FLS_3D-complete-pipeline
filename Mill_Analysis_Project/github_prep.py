#!/usr/bin/env python3
"""
GitHub Repository Preparation Tool for Mill Analysis Project
Professional utility for preparing the repository for GitHub upload.
"""

import os
import shutil
import sys
from pathlib import Path

class GitHubPrep:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.current_gitignore = self.repo_path / ".gitignore"
        self.enhanced_gitignore = self.repo_path / ".gitignore_enhanced"
        
    def analyze_repository_size(self) -> dict:
        """Analyze repository size and identify large files/directories."""
        analysis = {
            "large_files": [],
            "large_dirs": [],
            "total_size": 0,
            "ignored_size": 0
        }
        
        # Read current gitignore patterns
        gitignore_patterns = self._read_gitignore_patterns()
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip git directory
            if ".git" in root:
                continue
                
            for file in files:
                filepath = Path(root) / file
                try:
                    size = filepath.stat().st_size
                    analysis["total_size"] += size
                    
                    # Check if file should be ignored
                    if self._should_ignore(filepath, gitignore_patterns):
                        analysis["ignored_size"] += size
                    else:
                        # Flag files larger than 10MB
                        if size > 10 * 1024 * 1024:
                            analysis["large_files"].append({
                                "path": str(filepath.relative_to(self.repo_path)),
                                "size_mb": size / (1024 * 1024)
                            })
                except (OSError, PermissionError):
                    continue
        
        return analysis
    
    def _read_gitignore_patterns(self) -> list:
        """Read gitignore patterns from current .gitignore file."""
        patterns = []
        if self.current_gitignore.exists():
            with open(self.current_gitignore, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
        return patterns
    
    def _should_ignore(self, filepath: Path, patterns: list) -> bool:
        """Check if file should be ignored based on gitignore patterns."""
        relative_path = str(filepath.relative_to(self.repo_path))
        
        for pattern in patterns:
            # Simple pattern matching (basic implementation)
            if pattern.endswith('/') and pattern[:-1] in relative_path:
                return True
            elif pattern in relative_path or relative_path.endswith(pattern):
                return True
            elif pattern.startswith('*.') and relative_path.endswith(pattern[1:]):
                return True
        
        return False
    
    def backup_current_gitignore(self) -> bool:
        """Backup current .gitignore file."""
        if self.current_gitignore.exists():
            backup_path = self.repo_path / ".gitignore.backup"
            shutil.copy2(self.current_gitignore, backup_path)
            print(f"✅ Backed up current .gitignore to .gitignore.backup")
            return True
        return False
    
    def upgrade_gitignore(self) -> bool:
        """Replace current .gitignore with enhanced version."""
        if not self.enhanced_gitignore.exists():
            print("❌ Enhanced .gitignore file not found!")
            return False
        
        # Backup current
        self.backup_current_gitignore()
        
        # Replace with enhanced version
        shutil.copy2(self.enhanced_gitignore, self.current_gitignore)
        print("✅ Upgraded .gitignore with enhanced version")
        
        # Remove the temporary enhanced file
        self.enhanced_gitignore.unlink()
        print("✅ Cleaned up temporary files")
        
        return True
    
    def generate_repository_report(self) -> str:
        """Generate a comprehensive repository analysis report."""
        analysis = self.analyze_repository_size()
        
        report = []
        report.append("=" * 60)
        report.append("MILL ANALYSIS PROJECT - GITHUB PREPARATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Size analysis
        total_size_mb = analysis["total_size"] / (1024 * 1024)
        ignored_size_mb = analysis["ignored_size"] / (1024 * 1024)
        upload_size_mb = total_size_mb - ignored_size_mb
        
        report.append(f"📊 REPOSITORY SIZE ANALYSIS:")
        report.append(f"   Total repository size: {total_size_mb:.1f} MB")
        report.append(f"   Files ignored by .gitignore: {ignored_size_mb:.1f} MB")
        report.append(f"   Estimated upload size: {upload_size_mb:.1f} MB")
        report.append("")
        
        # GitHub limits
        if upload_size_mb > 1000:
            report.append("⚠️  WARNING: Repository > 1GB (GitHub soft limit)")
        elif upload_size_mb > 100:
            report.append("⚠️  NOTICE: Repository > 100MB (consider optimization)")
        else:
            report.append("✅ Repository size is GitHub-friendly")
        report.append("")
        
        # Large files
        if analysis["large_files"]:
            report.append("📁 LARGE FILES DETECTED:")
            for file_info in analysis["large_files"][:10]:  # Show top 10
                report.append(f"   • {file_info['path']} ({file_info['size_mb']:.1f} MB)")
            if len(analysis["large_files"]) > 10:
                report.append(f"   ... and {len(analysis['large_files']) - 10} more")
            report.append("")
        
        # Recommendations
        report.append("🚀 GITHUB UPLOAD RECOMMENDATIONS:")
        report.append("   1. Use enhanced .gitignore (run upgrade_gitignore())")
        report.append("   2. Verify large data files are properly ignored")
        report.append("   3. Consider Git LFS for files > 100MB")
        report.append("   4. Remove virtual environment before upload")
        report.append("   5. Ensure sensitive data is excluded")
        report.append("")
        
        return "\n".join(report)
    
    def clean_for_github(self) -> None:
        """Perform all cleanup operations for GitHub upload."""
        print("🧹 Preparing repository for GitHub upload...")
        print()
        
        # Generate and display report
        report = self.generate_repository_report()
        print(report)
        
        # Ask for confirmation
        response = input("Would you like to upgrade to the enhanced .gitignore? (y/n): ")
        if response.lower() in ['y', 'yes']:
            self.upgrade_gitignore()
            print()
            print("✅ Repository is now ready for GitHub upload!")
            print("💡 Next steps:")
            print("   1. Initialize git: git init")
            print("   2. Add files: git add .")
            print("   3. Commit: git commit -m 'Initial commit: Mill Analysis Project'")
            print("   4. Add remote: git remote add origin <your-repo-url>")
            print("   5. Push: git push -u origin main")
        else:
            print("ℹ️  .gitignore upgrade skipped")

def main():
    """Main function for command line usage."""
    prep = GitHubPrep()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "analyze":
            print(prep.generate_repository_report())
        elif command == "upgrade":
            prep.upgrade_gitignore()
        elif command == "clean":
            prep.clean_for_github()
        else:
            print("Usage: python github_prep.py [analyze|upgrade|clean]")
    else:
        # Interactive mode
        prep.clean_for_github()

if __name__ == "__main__":
    main()
