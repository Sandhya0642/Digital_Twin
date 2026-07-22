"""
motor_visual.py
An interactive, temperature-reactive motor graphic — a lightweight
stand-in for a full 3D digital twin viewer. Instead of only reading
numbers off a chart, the motor housing itself changes colour as the
temperature rises (cool teal -> amber -> hot red, with a pulsing glow
once it crosses into Critical territory), and the internal fan/rotor
spins faster or slower depending on live RPM.

Pure SVG + CSS, rendered through st.markdown(unsafe_allow_html=True) —
no extra dependency needed.
"""


def _temp_to_color(temperature: float) -> str:
    """Map temperature to an HSL colour: green/teal when cool, sliding
    through amber to red as it heats up — mirrors the health-status
    palette already used across the dashboard."""
    t = max(30.0, min(115.0, float(temperature)))
    # 30C -> hue 165 (teal/green) ... 100C+ -> hue 2 (red)
    hue = 165 - ((t - 30) / (100 - 30)) * 163
    hue = max(2, min(165, hue))
    return f"hsl({hue:.0f}, 68%, 42%)"


def _temp_to_glow(temperature: float, status: str) -> str:
    if status == "Critical":
        return "0 0 26px 6px rgba(193,68,60,0.55)"
    if status == "Warning":
        return "0 0 18px 4px rgba(217,164,65,0.40)"
    return "0 0 14px 2px rgba(46,154,93,0.25)"


def _rpm_to_duration(rpm: float) -> float:
    """Faster RPM => shorter spin-animation duration (faster looking spin)."""
    rpm = max(0.0, float(rpm))
    if rpm <= 0:
        return 999  # effectively stopped
    duration = 2200.0 / rpm  # seconds, tuned so ~1500 RPM ≈ 1.5s per revolution
    return max(0.25, min(6.0, duration))


def render_motor_visual(temperature: float, rpm: float, vibration: float, status: str, prediction: str) -> str:
    """Return an HTML/SVG string for the interactive motor. Caller should
    render it with st.markdown(html, unsafe_allow_html=True)."""

    body_color = _temp_to_color(temperature)
    glow = _temp_to_glow(temperature, status)
    spin_duration = _rpm_to_duration(rpm)
    is_critical = status == "Critical"
    is_warning = status == "Warning"

    # A little extra shake for high vibration, purely cosmetic.
    shake_amplitude = 0 if vibration < 4 else (1.2 if vibration < 7 else 2.6)
    shake_css = ""
    if shake_amplitude > 0:
        shake_css = f"""
        @keyframes motorShake {{
            0%, 100% {{ transform: translate(0, 0); }}
            25% {{ transform: translate({shake_amplitude}px, -{shake_amplitude/2}px); }}
            50% {{ transform: translate(-{shake_amplitude}px, {shake_amplitude/2}px); }}
            75% {{ transform: translate({shake_amplitude/2}px, {shake_amplitude}px); }}
        }}
        """

    status_badge_class = {
        "Healthy": "mv-badge-healthy",
        "Warning": "mv-badge-warning",
        "Critical": "mv-badge-critical",
    }.get(status, "mv-badge-warning")

    html = f"""
    <style>
    .mv-wrap {{
        background: #FFFFFF; border: 1px solid #E5EAE7; border-radius: 14px;
        padding: 22px 24px; box-shadow: 0 1px 2px rgba(23,34,29,0.04), 0 4px 16px rgba(23,34,29,0.05);
        display: flex; align-items: center; gap: 26px; flex-wrap: wrap;
    }}
    .mv-stage {{
        flex-shrink: 0; width: 220px; height: 150px; display: flex; align-items: center; justify-content: center;
        {"animation: motorShake 0.35s infinite;" if shake_amplitude > 0 else ""}
    }}
    .mv-housing {{ transition: filter 0.6s ease; filter: drop-shadow({glow}); }}
    .mv-body-fill {{ transition: fill 0.8s ease; fill: {body_color}; }}
    .mv-rotor {{
        transform-origin: 170px 75px;
        animation: mvSpin {spin_duration}s linear infinite;
    }}
    @keyframes mvSpin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    {shake_css}
    .mv-info {{ flex: 1 1 220px; min-width: 200px; }}
    .mv-title {{ font-family:'Poppins',sans-serif; font-weight:600; font-size:15px; color:#17221D; margin-bottom:6px; }}
    .mv-sub {{ font-family:'Inter',sans-serif; font-size:12.5px; color:#8B9A92; margin-bottom:12px; }}
    .mv-badge {{
        display:inline-block; padding:5px 14px; border-radius:999px; font-family:'Poppins',sans-serif;
        font-weight:600; font-size:12.5px; margin-right:8px;
    }}
    .mv-badge-healthy {{ background:#E4F3EB; color:#1E7A4C; }}
    .mv-badge-warning {{ background:#FDF1DC; color:#A8681B; }}
    .mv-badge-critical {{ background:#FBE6E6; color:#B3261E; }}
    .mv-legend {{ display:flex; gap:14px; margin-top:12px; flex-wrap:wrap; }}
    .mv-legend-item {{ display:flex; align-items:center; gap:6px; font-size:11.5px; color:#4A5A52; font-family:'Inter',sans-serif; }}
    .mv-legend-swatch {{ width:11px; height:11px; border-radius:3px; }}
    </style>
    <div class="mv-wrap">
        <div class="mv-stage">
            <svg class="mv-housing" width="220" height="150" viewBox="0 0 220 150" xmlns="http://www.w3.org/2000/svg">
                <!-- mounting feet -->
                <rect x="35" y="128" width="26" height="12" rx="2" fill="#C9D2CC"/>
                <rect x="150" y="128" width="26" height="12" rx="2" fill="#C9D2CC"/>
                <!-- cooling fins on housing -->
                <g opacity="0.35">
                    <rect x="40" y="35" width="4" height="70" fill="#0B140F"/>
                    <rect x="52" y="35" width="4" height="70" fill="#0B140F"/>
                    <rect x="64" y="35" width="4" height="70" fill="#0B140F"/>
                    <rect x="76" y="35" width="4" height="70" fill="#0B140F"/>
                    <rect x="88" y="35" width="4" height="70" fill="#0B140F"/>
                    <rect x="100" y="35" width="4" height="70" fill="#0B140F"/>
                    <rect x="112" y="35" width="4" height="70" fill="#0B140F"/>
                </g>
                <!-- main cylindrical body -->
                <rect class="mv-body-fill" x="30" y="30" width="100" height="80" rx="14"/>
                <rect x="30" y="30" width="100" height="26" rx="14" fill="#FFFFFF" opacity="0.18"/>
                <!-- shaft coupling -->
                <rect class="mv-body-fill" x="128" y="58" width="24" height="24" rx="4"/>
                <!-- end bell / fan housing -->
                <circle class="mv-body-fill" cx="170" cy="70" r="42"/>
                <circle cx="170" cy="70" r="42" fill="none" stroke="#0B140F" stroke-opacity="0.18" stroke-width="2"/>
                <circle cx="170" cy="70" r="30" fill="#F6F8F6" opacity="0.85"/>
                <!-- rotor / fan blades (this spins) -->
                <g class="mv-rotor">
                    <ellipse cx="170" cy="70" rx="26" ry="7" fill="#17221D" opacity="0.55"/>
                    <ellipse cx="170" cy="70" rx="7" ry="26" fill="#17221D" opacity="0.55"/>
                    <ellipse cx="170" cy="70" rx="18" ry="18" fill="#4A5A52" opacity="0.35" transform="rotate(45 170 70)"/>
                    <circle cx="170" cy="70" r="6" fill="#17221D"/>
                </g>
                <!-- nameplate -->
                <rect x="46" y="66" width="66" height="16" rx="3" fill="#FFFFFF" opacity="0.85"/>
                <text x="79" y="77.5" font-family="JetBrains Mono, monospace" font-size="8" fill="#17221D" text-anchor="middle">MOTOR-01</text>
            </svg>
        </div>
        <div class="mv-info">
            <div class="mv-title">Interactive Motor Twin</div>
            <div class="mv-sub">Housing colour tracks live temperature · rotor speed tracks live RPM</div>
            <span class="mv-badge {status_badge_class}">{status}</span>
            <span class="mv-badge" style="background:#F6F8F6; color:#4A5A52;">{prediction}</span>
            <div class="mv-legend">
                <div class="mv-legend-item"><div class="mv-legend-swatch" style="background:hsl(165,68%,42%);"></div> Cool / Healthy</div>
                <div class="mv-legend-item"><div class="mv-legend-swatch" style="background:hsl(80,68%,42%);"></div> Warming</div>
                <div class="mv-legend-item"><div class="mv-legend-swatch" style="background:hsl(2,68%,42%);"></div> Hot / Critical</div>
            </div>
        </div>
    </div>
    """
    return html
