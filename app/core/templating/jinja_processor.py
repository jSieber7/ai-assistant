"""
Jinja template processing system for multi-writer/checker system
"""
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
import asyncio
import re

class JinjaProcessor:
    """Processes content using Jinja templates"""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )
        
        # Add custom filters
        self.env.filters['wordcount'] = self._wordcount_filter
        self.env.filters['reading_time'] = self._reading_time_filter
        self.env.filters['seo_slug'] = self._seo_slug_filter
    
    async def render_content(
        self, 
        template_name: str, 
        content_data: Dict[str, Any],
        additional_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Render content using Jinja template"""
        
        try:
            # Load template
            template = self.env.get_template(template_name)
            
            # Prepare context
            context = {
                "content": content_data,
                **(additional_context or {})
            }
            
            # Render template
            rendered_content = template.render(**context)
            
            # Generate metadata
            metadata = self._generate_metadata(rendered_content, content_data)
            
            return {
                "success": True,
                "rendered_content": rendered_content,
                "template_used": template_name,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "template_used": template_name
            }
    
    def _wordcount_filter(self, text: str) -> int:
        """Custom filter for word count"""
        return len(text.split())
    
    def _reading_time_filter(self, text: str) -> int:
        """Custom filter for estimated reading time (minutes)"""
        word_count = len(text.split())
        return max(1, round(word_count / 200))  # Assume 200 words per minute
    
    def _seo_slug_filter(self, text: str) -> str:
        """Custom filter for SEO-friendly URLs"""
        # Convert to lowercase and replace spaces with hyphens
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    def _generate_metadata(self, rendered_content: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for rendered content"""
        return {
            "word_count": len(rendered_content.split()),
            "character_count": len(rendered_content),
            "paragraph_count": len([p for p in rendered_content.split('\n\n') if p.strip()]),
            "reading_time_minutes": max(1, round(len(rendered_content.split()) / 200)),
            "has_headings": '<h1' in rendered_content or '<h2' in rendered_content,
            "has_links": '<a href=' in rendered_content,
            "has_images": '<img' in rendered_content,
            "original_writer": content_data.get("original_content", {}).get("writer_id"),
            "quality_score": content_data.get("overall_score", 0),
            "template_applied": content_data.get("template_used", "")
        }
    
    async def batch_render(
        self, 
        content_list: List[Dict[str, Any]], 
        template_name: str,
        additional_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Render multiple content items using the same template"""
        tasks = [
            self.render_content(template_name, content, additional_context)
            for content in content_list
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Create default templates directory and templates
def create_default_templates(template_dir: Path):
    """Create default Jinja templates for the system"""
    template_dir.mkdir(exist_ok=True)
    
    # HTML article template
    article_html = """<!DOCTYPE html>
<html>
<head>
    <title>{{ content.title | default("Untitled Article") }}</title>
    <meta name="description" content="{{ content.summary | default("") }}">
    <meta name="author" content="{{ content.original_content.writer_id | default("AI Writer") }}">
    <meta name="word-count" content="{{ content.best_improved_version.content | wordcount }}">
    <meta name="reading-time" content="{{ content.best_improved_version.content | reading_time }}">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .sources { background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .editor-notes { background: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .metadata { background: #e9ecef; padding: 10px; margin: 20px 0; border-radius: 5px; font-size: 0.9em; }
    </style>
</head>
<body>
    <article>
        <h1>{{ content.title | default("Untitled Article") }}</h1>
        
        {% if content.original_content.sources_used %}
        <div class="sources">
            <h3>Sources:</h3>
            <ul>
            {% for source in content.original_content.sources_used %}
                <li><a href="{{ source }}">{{ source }}</a></li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <div class="content">
            {{ content.best_improved_version.content | safe }}
        </div>
        
        {% if content.aggregated_feedback.recommendations %}
        <div class="editor-notes">
            <h3>Editor Notes:</h3>
            <ul>
            {% for recommendation in content.aggregated_feedback.recommendations %}
                <li>{{ recommendation }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <div class="metadata">
            <p>Quality Score: {{ content.overall_score }}/100</p>
            <p>Writer: {{ content.original_content.specialty | title }} Writer</p>
            <p>Reading Time: {{ content.best_improved_version.content | reading_time }} minutes</p>
        </div>
    </article>
</body>
</html>"""
    
    # Markdown blog post template
    blog_post_md = """# {{ content.title | default("Untitled Post") }}

**Writer:** {{ content.original_content.specialty | title }} Writer  
**Quality Score:** {{ content.overall_score }}/100  
**Reading Time:** {{ content.best_improved_version.content | reading_time }} minutes

---

{{ content.best_improved_version.content }}

---

## Sources

{% if content.original_content.sources_used %}
{% for source in content.original_content.sources_used %}
- [Source]({{ source }})
{% endfor %}
{% else %}
*No external sources cited*
{% endif %}

## Editor Notes

{% if content.aggregated_feedback.recommendations %}
{% for recommendation in content.aggregated_feedback.recommendations %}
- {{ recommendation }}
{% endfor %}
{% else %}
*No specific editor notes for this content*
{% endif %}"""
    
    # Write templates to files
    with open(template_dir / "article.html.jinja", "w") as f:
        f.write(article_html)
    
    with open(template_dir / "blog-post.md.jinja", "w") as f:
        f.write(blog_post_md)