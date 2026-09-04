"""Unit tests for crawler_utils module."""

import pytest
from common.crawler_utils import (
    clean_html_text,
    clean_wikitext,
    compute_script_fraction,
    estimate_tokens,
)


def test_clean_html_text_removes_tags_and_scripts():
    raw_html = """
    <html>
      <head><title>Test Page</title><script>alert('xss');</script></head>
      <body>
        <nav><a href="/home">Home</a></nav>
        <article>
          <h1>मुख्य समाचार</h1>
          <p>यह भारत की राजधानी नई दिल्ली में आयोजित एक महत्वपूर्ण सम्मेलन की खबर है।</p>
          <p>सम्मेलन में पर्यावरण संरक्षण और सतत विकास पर विस्तार से चर्चा की गई।</p>
        </article>
        <footer><p>Copyright © 2026 All Rights Reserved</p></footer>
      </body>
    </html>
    """
    cleaned = clean_html_text(raw_html)
    assert "alert" not in cleaned
    assert "<nav>" not in cleaned
    assert "मुख्य समाचार" in cleaned
    assert "पर्यावरण संरक्षण" in cleaned


def test_clean_wikitext_removes_templates_and_tables():
    raw_wikitext = """
    {{Infobox writer | name = प्रेमचंद | birth_date = 1880 }}
    <!-- This is a comment -->
    == जीवन परिचय ==
    मुंशी प्रेमचंद हिंदी और उर्दू के सर्वाधिक लोकप्रिय उपन्यासकार, कहानीकार एवं विचारक थे।
    {| class="wikitable"
    |-
    ! वर्ष !! रचना
    |-
    | 1936 || गोदान
    |}
    [[Category:हिंदी साहित्यकार]]
    """
    cleaned = clean_wikitext(raw_wikitext)
    assert "Infobox" not in cleaned
    assert "comment" not in cleaned
    assert "Category" not in cleaned
    assert "{|" not in cleaned
    assert "मुंशी प्रेमचंद हिंदी और उर्दू" in cleaned


def test_compute_script_fraction_hindi():
    hindi_text = "यह एक मानक हिंदी वाक्य है जो देवनागरी लिपि में लिखा गया है।"
    in_s, tot, frac = compute_script_fraction(hindi_text, "hindi")
    assert frac > 0.90
    assert in_s > 0


def test_compute_script_fraction_assamese():
    asm_text = "এইটো এটা অসমীয়া ভাষাৰ বাক্য আৰু ইয়াত অসমীয়া বৰ্ণমালা ব্যৱহাৰ কৰা হৈছে।"
    in_s, tot, frac = compute_script_fraction(asm_text, "assamese")
    assert frac > 0.90
    assert in_s > 0


def test_estimate_tokens():
    text = "भारत एक विशाल और विविधताओं से भरा हुआ देश है।"
    tokens = estimate_tokens(text)
    assert tokens >= len(text.split())
