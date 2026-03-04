from m3.attr_contract import attr_del, attr_get_optional, attr_get_required, attr_has, attr_set, guard_context, guard_eval, guard_step
import logging
import os
import ast
from collections import defaultdict

def get_context(node):
    curr = attr_get_optional(node, 'parent', None)
    while curr is not None:
        if isinstance(curr, ast.FunctionDef):
            return f"Function: {curr.name}"
        elif isinstance(curr, ast.ClassDef):
            return f"Class: {curr.name}"
        curr = attr_get_optional(curr, 'parent', None)
    return "Module level"

repo_dir = r"c:\Users\조승준\OneDrive\바탕 화면\M3-main\M3-main"
results = defaultdict(list)

for root, _, files in os.walk(repo_dir):
    for file in files:
        if file.endswith('.py') and file != 'find_exceptions.py':
            path = os.path.join(root, file)
            with guard_context(ctx='build_exception_report.py:41', catch_base=False) as __m3_guard_23_12:
                with open(path, 'r', encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source, filename=path)
                
                # Assign parents
                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        child.parent = node
                        
                for node in ast.walk(tree):
                    if isinstance(node, ast.Try):
                        for handler in node.handlers:
                            if attr_get_optional(handler.type, 'id', None) == 'Exception':
                                if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                                    ctx = get_context(node)
                                    rel_path = os.path.relpath(path, repo_dir)
                                    results[rel_path].append({'line': handler.lineno, 'context': ctx})

            if __m3_guard_23_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

with open(os.path.join(repo_dir, "swallowed_exceptions_report.md"), "w", encoding="utf-8") as f:
    f.write("# 🚨 Swallowed Exceptions Deep Dive Report\n\n")
    f.write("This report identifies every `except Exception: pass` anti-pattern in the codebase.\n\n")
    
    total = sum(len(items) for items in results.values())
    f.write(f"**Total Swallowed Exceptions Found:** {total}\n\n")
    
    sorted_files = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
    for file, items in sorted_files:
        f.write(f"## `{file}` ({len(items)} items)\n")
        # Group by context
        ctx_groups = defaultdict(list)
        for item in items:
            ctx_groups[item['context']].append(item['line'])
        
        for ctx, lines in ctx_groups.items():
            line_str = ", ".join(map(str, sorted(lines)))
            f.write(f"- **{ctx}** (Lines: {line_str})\n")
        f.write("\n")

print("Report generated successfully.")
