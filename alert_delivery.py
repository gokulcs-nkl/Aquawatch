"""
alert_delivery.py — Free WhatsApp & Email Alert Delivery
============================================================
Uses pywhatkit (WhatsApp Web automation) — completely FREE, no API keys.
Also supports Email alerts via SMTP (Gmail / any provider).

WhatsApp: Uses your own WhatsApp Web session (must be logged in on browser).
Email:    Uses SMTP — set env vars or pass credentials directly.

Usage:
    from alert_delivery import send_whatsapp_alert, send_email_alert
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════
# WhatsApp Alert via pywhatkit (FREE — uses WhatsApp Web)
# ═══════════════════════════════════════════════════════════════════════════

def send_whatsapp_alert(to_number: str, message: str, instant: bool = True) -> str:
    """
    Send WhatsApp message using pywhatkit (WhatsApp Web automation).

    Prerequisites:
        - WhatsApp Web must be logged in on your default browser
        - pip install pywhatkit

    Args:
        to_number: Phone number with country code (e.g. "+919876543210")
        message:   Alert text to send
        instant:   If True, sends instantly. If False, schedules 2 min ahead.

    Returns:
        Status string confirming the send attempt.
    """
    import pywhatkit as kit

    if not to_number.startswith("+"):
        to_number = f"+{to_number}"

    if instant:
        # sendwhatmsg_instantly opens WhatsApp Web and sends immediately
        kit.sendwhatmsg_instantly(
            phone_no=to_number,
            message=message,
            wait_time=15,       # seconds to wait for WhatsApp Web to load
            tab_close=True,     # close browser tab after sending
        )
        return f"whatsapp_instant_sent_to_{to_number}"
    else:
        # Schedule 2 minutes from now
        now = datetime.now() + timedelta(minutes=2)
        kit.sendwhatmsg(
            phone_no=to_number,
            message=message,
            time_hour=now.hour,
            time_min=now.minute,
            wait_time=15,
            tab_close=True,
        )
        return f"whatsapp_scheduled_{now.hour}:{now.minute:02d}_to_{to_number}"


def send_whatsapp_group_alert(group_id: str, message: str) -> str:
    """
    Send WhatsApp message to a group using pywhatkit.

    Args:
        group_id: WhatsApp group invite link ID (the part after chat.whatsapp.com/)
        message:  Alert text to send

    Returns:
        Status string.
    """
    import pywhatkit as kit

    now = datetime.now() + timedelta(minutes=2)
    kit.sendwhatmsg_to_group(
        group_id=group_id,
        message=message,
        time_hour=now.hour,
        time_min=now.minute,
        wait_time=15,
        tab_close=True,
    )
    return f"whatsapp_group_scheduled_{group_id}"


# ═══════════════════════════════════════════════════════════════════════════
# Email Alert via SMTP (FREE with Gmail / any email provider)
# ═══════════════════════════════════════════════════════════════════════════

# Default credentials from environment (optional)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # Gmail: use App Password


def send_email_alert(
    to_email: str,
    subject: str,
    message: str,
    from_email: str = "",
    smtp_password: str = "",
    smtp_server: str = "",
    smtp_port: int = 0,
) -> str:
    """
    Send email alert via SMTP.

    Args:
        to_email:     Recipient email address
        subject:      Email subject line
        message:      Email body (plain text)
        from_email:   Sender email (defaults to SMTP_EMAIL env var)
        smtp_password: SMTP password (defaults to SMTP_PASSWORD env var)
        smtp_server:  SMTP server (defaults to smtp.gmail.com)
        smtp_port:    SMTP port (defaults to 587)

    Returns:
        Status string confirming send.
    """
    sender = from_email or SMTP_EMAIL
    password = smtp_password or SMTP_PASSWORD
    server = smtp_server or SMTP_SERVER
    port = smtp_port or SMTP_PORT

    if not sender or not password:
        raise ValueError(
            "Email credentials not configured. Set SMTP_EMAIL and SMTP_PASSWORD "
            "environment variables, or pass from_email and smtp_password."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    # Plain text version
    msg.attach(MIMEText(message, "plain"))

    # HTML version (styled)
    html_body = f"""
    <html>
    <body style="font-family: Segoe UI, sans-serif; padding: 20px;">
        <div style="background: #f0f9ff; border-left: 5px solid #0369a1;
            border-radius: 8px; padding: 20px; max-width: 600px;">
            <h2 style="color: #0369a1; margin-top: 0;">🚨 AquaWatch Alert</h2>
            <pre style="font-size: 14px; line-height: 1.6; color: #334155;
                white-space: pre-wrap;">{message}</pre>
            <hr style="border-color: #e2e8f0;">
            <p style="font-size: 12px; color: #94a3b8;">
                Sent by AquaWatch Water Risk Monitor · {datetime.now().strftime('%d %b %Y %H:%M UTC')}
            </p>
        </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(server, port) as smtp:
        smtp.starttls()
        smtp.login(sender, password)
        smtp.send_message(msg)

    return f"email_sent_to_{to_email}"


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Build alert message from risk data
# ═══════════════════════════════════════════════════════════════════════════

def build_alert_message(
    lat: float,
    lon: float,
    risk_score: float,
    risk_level: str,
    risk_emoji: str,
    who_severity: str,
    cells_per_ml: int,
    trend: str,
    trend_emoji: str,
    water_temp: float,
    wind_speed: float,
    confidence: str,
) -> str:
    """Build a formatted alert message from risk assessment data."""
    return (
        f"🚨 AquaWatch Bloom Risk Alert\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Location: {lat:.4f}, {lon:.4f}\n"
        f"⚠️ Risk Level: {risk_emoji} {risk_level}\n"
        f"📊 Risk Score: {risk_score:.1f}/100\n"
        f"🏥 WHO: {who_severity.replace('_', ' ').title()}\n"
        f"🔬 Est. Cells: {cells_per_ml:,}/mL\n"
        f"📈 Trend: {trend_emoji} {trend}\n"
        f"🌡 Water Temp: {water_temp:.1f}°C\n"
        f"💨 Wind: {wind_speed:.0f} km/h\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.now().strftime('%d %b %Y %H:%M UTC')}\n"
        f"Confidence: {confidence}"
    )


if __name__ == "__main__":
    # Quick test
    print("Alert delivery module loaded.")
    print("Available functions:")
    print("  - send_whatsapp_alert(phone, message)")
    print("  - send_whatsapp_group_alert(group_id, message)")
    print("  - send_email_alert(to_email, subject, message)")
    print("  - build_alert_message(...)")
    print()
    print("WhatsApp: Requires WhatsApp Web logged in on browser.")
    print("Email: Set SMTP_EMAIL and SMTP_PASSWORD env vars (Gmail App Password).")
