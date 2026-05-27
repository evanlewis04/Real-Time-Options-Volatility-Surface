"""Keyboard shortcut and command-palette bridge for the Streamlit dashboard."""

from __future__ import annotations

import json

import streamlit as st


def render_keyboard_layer(*, page_labels: list[str], symbols: list[str]) -> None:
    """Inject a small parent-document keyboard controller."""
    payload = json.dumps({"pages": page_labels, "symbols": symbols[:24]})
    st.iframe(
        f"""
        <script>
        (() => {{
            const data = {payload};
            const doc = window.parent.document;
            if (doc.__volSurfaceKeyboardLayer) return;
            doc.__volSurfaceKeyboardLayer = true;

            const style = doc.createElement("style");
            style.textContent = `
                #vs-command-palette {{
                    position: fixed;
                    inset: 0;
                    z-index: 999999;
                    display: none;
                    align-items: flex-start;
                    justify-content: center;
                    padding-top: 9vh;
                    background: rgba(11, 13, 16, 0.72);
                }}
                #vs-command-palette.open {{ display: flex; }}
                .vs-palette-panel {{
                    width: min(720px, calc(100vw - 32px));
                    border: 1px solid rgba(255,255,255,0.14);
                    border-radius: 8px;
                    background: #11141A;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 28px 80px rgba(0,0,0,0.45);
                    color: #F4F7FB;
                    font-family: Inter, Segoe UI, sans-serif;
                    overflow: hidden;
                }}
                .vs-palette-search {{
                    width: 100%;
                    box-sizing: border-box;
                    border: 0;
                    border-bottom: 1px solid rgba(255,255,255,0.12);
                    background: #0B0D10;
                    color: #F4F7FB;
                    font: 700 15px Inter, Segoe UI, sans-serif;
                    outline: none;
                    padding: 14px 16px;
                }}
                .vs-palette-list {{ max-height: 420px; overflow: auto; padding: 8px; }}
                .vs-palette-item {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    min-height: 34px;
                    border-radius: 6px;
                    padding: 0 10px;
                    color: #B6C0CC;
                    font-size: 12px;
                    font-weight: 700;
                }}
                .vs-palette-item:hover, .vs-palette-item.active {{
                    background: rgba(245,165,36,0.08);
                    color: #F5A524;
                }}
                .vs-palette-item code {{
                    color: #7F8A98;
                    font-family: "JetBrains Mono", Consolas, monospace;
                    font-size: 11px;
                }}
            `;
            doc.head.appendChild(style);

            const overlay = doc.createElement("div");
            overlay.id = "vs-command-palette";
            overlay.innerHTML = `
                <div class="vs-palette-panel" role="dialog" aria-label="Command palette">
                    <input class="vs-palette-search" placeholder="Search symbols, pages, and actions..." />
                    <div class="vs-palette-list"></div>
                </div>
            `;
            doc.body.appendChild(overlay);

            const input = overlay.querySelector(".vs-palette-search");
            const list = overlay.querySelector(".vs-palette-list");
            const commands = [
                ...data.pages.map((label, index) => ({{
                    label: `F${{index + 1}} / ${{index === 9 ? 0 : index + 1}} -> ${{label}}`,
                    hint: "page",
                    run: () => clickTab(index)
                }})),
                ...data.symbols.map((symbol) => ({{
                    label: `Jump symbol ${{symbol}}`,
                    hint: "symbol",
                    run: () => focusSymbol(symbol)
                }})),
                {{ label: "Toggle 3D surface", hint: "view", run: () => clickByLabel("3D surface") }},
                {{ label: "Refresh data", hint: "action", run: refresh }},
                {{ label: "Copy current view as PNG", hint: "export", run: copyCurrentViewPng }},
                {{ label: "Keyboard shortcuts", hint: "help", run: () => showHelp() }},
            ];

            function render(query = "") {{
                const q = query.trim().toLowerCase();
                const rows = commands
                    .filter((command) => !q || command.label.toLowerCase().includes(q) || command.hint.includes(q))
                    .slice(0, 12);
                list.innerHTML = rows.map((command, index) =>
                    `<div class="vs-palette-item ${{index === 0 ? "active" : ""}}" data-index="${{commands.indexOf(command)}}">` +
                    `<span>${{command.label}}</span><code>${{command.hint}}</code></div>`
                ).join("");
            }}

            function openPalette(seed = "") {{
                overlay.classList.add("open");
                render(seed);
                input.value = seed;
                setTimeout(() => input.focus(), 0);
            }}
            function closePalette() {{
                overlay.classList.remove("open");
            }}
            function clickTab(index) {{
                const tabs = doc.querySelectorAll('[role="tab"]');
                tabs[index]?.click();
                syncHeaderTabs();
            }}
            function syncHeaderTabs() {{
                const tabs = [...doc.querySelectorAll('[role="tab"]')];
                const selectedIndex = tabs.findIndex((tab) => tab.getAttribute("aria-selected") === "true");
                doc.querySelectorAll("[data-vs-tab-index]").forEach((item) => {{
                    const active = Number(item.dataset.vsTabIndex) === selectedIndex;
                    item.classList.toggle("active", active);
                    if (active) {{
                        item.setAttribute("aria-current", "page");
                    }} else {{
                        item.removeAttribute("aria-current");
                    }}
                }});
            }}
            function refresh() {{
                [...doc.querySelectorAll("button")]
                    .find((button) => button.textContent.trim() === "Refresh data")
                    ?.click();
            }}
            function focusSymbol(seed = "") {{
                const search = [...doc.querySelectorAll("input")]
                    .find((node) => (node.placeholder || "").includes("Search ticker"));
                if (search) {{
                    search.focus();
                    if (seed) {{
                        search.value = seed;
                        search.dispatchEvent(new Event("input", {{ bubbles: true }}));
                    }}
                }}
            }}
            function clickByLabel(label) {{
                const labelNode = [...doc.querySelectorAll("label")]
                    .find((node) => node.textContent.includes(label));
                labelNode?.click();
            }}
            function showHelp() {{
                openPalette("Keyboard");
            }}
            async function copyCurrentViewPng() {{
                const node = doc.querySelector(".stApp") || doc.body;
                const width = window.parent.innerWidth;
                const height = window.parent.innerHeight;
                const styles = [...doc.styleSheets].map((sheet) => {{
                    try {{
                        return [...sheet.cssRules].map((rule) => rule.cssText).join("\\n");
                    }} catch (_error) {{
                        return "";
                    }}
                }}).join("\\n");
                const clone = node.cloneNode(true);
                clone.querySelectorAll("script, iframe").forEach((item) => item.remove());
                const markup = new XMLSerializer().serializeToString(clone);
                const svg =
                    `<svg xmlns="http://www.w3.org/2000/svg" width="${{width}}" height="${{height}}">` +
                    `<foreignObject width="100%" height="100%">` +
                    `<style>${{styles}}</style>${{markup}}` +
                    `</foreignObject></svg>`;
                const image = new Image();
                image.onload = async () => {{
                    const canvas = doc.createElement("canvas");
                    canvas.width = width;
                    canvas.height = height;
                    canvas.getContext("2d").drawImage(image, 0, 0);
                    canvas.toBlob(async (blob) => {{
                        try {{
                            await window.parent.navigator.clipboard.write([
                                new ClipboardItem({{ "image/png": blob }})
                            ]);
                        }} catch (_error) {{
                            await window.parent.navigator.clipboard.writeText("PNG clipboard capture was blocked by the browser.");
                        }}
                    }}, "image/png");
                }};
                image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
            }}

            overlay.addEventListener("click", (event) => {{
                if (event.target === overlay) closePalette();
                const item = event.target.closest(".vs-palette-item");
                if (item) {{
                    const command = commands[Number(item.dataset.index)];
                    closePalette();
                    command?.run();
                }}
            }});
            doc.addEventListener("click", (event) => {{
                const navItem = event.target.closest("[data-vs-tab-index]");
                if (!navItem) return;
                event.preventDefault();
                clickTab(Number(navItem.dataset.vsTabIndex));
            }});
            input.addEventListener("input", () => render(input.value));
            input.addEventListener("keydown", (event) => {{
                if (event.key === "Enter") {{
                    const item = list.querySelector(".vs-palette-item.active");
                    const command = item ? commands[Number(item.dataset.index)] : null;
                    closePalette();
                    command?.run();
                }}
                if (event.key === "Escape") closePalette();
            }});
            doc.addEventListener("keydown", (event) => {{
                const target = event.target;
                const editing = target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
                if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {{
                    event.preventDefault();
                    openPalette();
                    return;
                }}
                if (event.key === "?" && !editing) {{
                    event.preventDefault();
                    showHelp();
                    return;
                }}
                if (event.key === "/" && !editing) {{
                    event.preventDefault();
                    focusSymbol();
                    return;
                }}
                if (event.key.toLowerCase() === "r" && !editing) {{
                    event.preventDefault();
                    refresh();
                    return;
                }}
                const functionKeyMatch = event.key.match(/^F([1-9]|10)$/);
                if (functionKeyMatch && !editing) {{
                    event.preventDefault();
                    clickTab(Number(functionKeyMatch[1]) - 1);
                    return;
                }}
                if (/^[0-9]$/.test(event.key) && !editing) {{
                    event.preventDefault();
                    clickTab(event.key === "0" ? 9 : Number(event.key) - 1);
                }}
            }});
            const observer = new MutationObserver(syncHeaderTabs);
            observer.observe(doc.body, {{
                attributes: true,
                childList: true,
                subtree: true,
                attributeFilter: ["aria-selected", "class"]
            }});
            window.parent.setInterval(syncHeaderTabs, 1000);
            render();
            syncHeaderTabs();
        }})();
        </script>
        """,
        height=1,
    )
