from typing import Dict

HEADER_STYLES: Dict[str, Dict] = {
    "container": {
        "padding": "0px",
        "display": "grid",
        "margin": "0!important",
        "background-color": "#FEF8F4",  # Vibrant orange, aligns with primaryColor
        "max-width": "100vw"
    },
    "icon": {
        "color": "#FEF8F4",  # Soft off-white, aligns with backgroundColor
        "font-size": "14px"
    },
    "nav-link": {
        "font-size": "14px",
        "text-align": "center",
        "margin": "auto",
        "background-color": "#FFA580",  # Lighter shade of orange, aligns with secondaryBackgroundColor
        "height": "30px",
        "width": "7rem",
        "color": "#3B3B3B",  # Dark gray, aligns with textColor
        "border-radius": "5px"
    },
    "nav-link-selected": {
        "background-color": "#FF8A56",  # Slightly darker than the primary color to give a selected feel
        "font-weight": "300",
        "color": "#FEF8F4",  # Soft off-white, aligns with backgroundColor
        "border": "1px solid #FF9B50"  # Border with a shade of orange in between primary and secondary
    }
}
