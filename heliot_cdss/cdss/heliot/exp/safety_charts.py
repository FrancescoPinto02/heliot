import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set font to avoid missing character warnings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']

def create_professional_gauge():
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.8, 1.4)
    ax.set_aspect('equal')
    
    # Professional color scheme
    colors = {
        'critical': '#E53E3E',    # Professional red
        'warning': '#DD6B20',     # Professional orange  
        'good': '#38A169',        # Professional green
        'excellent': '#2D3748',   # Dark gray for needle
        'background': '#F7FAFC'   # Light background
    }
    
    # Define gauge zones
    zones = [
        (0, 60, colors['critical'], 'CRITICAL'),
        (60, 85, colors['warning'], 'WARNING'), 
        (85, 95, colors['good'], 'GOOD'),
        (95, 100, colors['good'], 'EXCELLENT')
    ]
    
    # Draw gauge background with subtle shadow
    shadow = Wedge((0.02, -0.02), 1.1, 0, 180, 
                   facecolor='gray', alpha=0.1)
    ax.add_patch(shadow)
    
    # Draw main gauge segments
    for start, end, color, label in zones:
        start_angle = 180 - (start/100 * 180)
        end_angle = 180 - (end/100 * 180)
        
        # Main segment
        wedge = Wedge((0, 0), 1, end_angle, start_angle, 
                     facecolor=color, alpha=0.8, 
                     edgecolor='white', linewidth=2)
        ax.add_patch(wedge)
        
        # Inner ring for depth
        inner_wedge = Wedge((0, 0), 0.7, end_angle, start_angle, 
                           facecolor=color, alpha=0.4)
        ax.add_patch(inner_wedge)
    
    # HELIOT score (99.0% - in the excellent zone)
    score = 99.0
    
    # Draw needle with professional styling
    angle = np.radians(180 - (score/100 * 180))
    needle_length = 0.85
    needle_x = needle_length * np.cos(angle)
    needle_y = needle_length * np.sin(angle)
    
    # Needle shadow
    shadow_x = needle_x + 0.02
    shadow_y = needle_y - 0.02
    ax.plot([0, shadow_x], [0, shadow_y], color='gray', alpha=0.3, linewidth=4)
    
    # Main needle
    ax.plot([0, needle_x], [0, needle_y], color=colors['excellent'], linewidth=6)
    ax.plot([0, needle_x], [0, needle_y], color='white', linewidth=2)
    
    # Needle tip
    tip = Circle((needle_x, needle_y), 0.04, facecolor=colors['excellent'], 
                edgecolor='white', linewidth=2, zorder=10)
    ax.add_patch(tip)
    
    # Center hub with gradient effect
    center_outer = Circle((0, 0), 0.12, facecolor=colors['excellent'], 
                         edgecolor='white', linewidth=3, zorder=8)
    center_inner = Circle((0, 0), 0.06, facecolor='white', zorder=9)
    ax.add_patch(center_outer)
    ax.add_patch(center_inner)
    
    # Score display with professional styling
    score_box = FancyBboxPatch((-.25, -0.25), 0.5, 0.15, 
                              boxstyle="round,pad=0.02", 
                              facecolor='white', 
                              edgecolor=colors['excellent'],
                              linewidth=2)
    ax.add_patch(score_box)
    
    ax.text(0, -0.175, f'{score:.1f}%', ha='center', va='center', 
            fontsize=18, fontweight='bold', color=colors['excellent'])
    
    # Title
    ax.text(0, 1.25, 'HELIOT Safety Performance', ha='center', va='center', 
            fontsize=14, fontweight='bold', color=colors['excellent'])
    
    # Zone labels OUTSIDE the semicircle with connecting lines
    zone_labels = [
        (30, 'CRITICAL', colors['critical'], 'left'),    # 30% position
        (72.5, 'WARNING', colors['warning'], 'center'),  # 72.5% position  
        (90, 'GOOD', colors['good'], 'center'),          # 90% position
        (97.5, 'EXCELLENT', colors['good'], 'right')     # 97.5% position
    ]
    
    for position, label, color, align in zone_labels:
        # Calculate angle for the middle of each zone
        angle_deg = 180 - (position/100 * 180)
        angle_rad = np.radians(angle_deg)
        
        # Point on the gauge circumference
        gauge_x = 1.0 * np.cos(angle_rad)
        gauge_y = 1.0 * np.sin(angle_rad)
        
        # Label position outside - increased distance for EXCELLENT
        if label == 'EXCELLENT':
            label_distance = 1.55  # Maggiore distanza per EXCELLENT
        else:
            label_distance = 1.35
            
        label_x = label_distance * np.cos(angle_rad)
        label_y = label_distance * np.sin(angle_rad)
        
        # Draw connecting line
        ax.plot([gauge_x, label_x], [gauge_y, label_y], 
                color=color, linewidth=2, alpha=0.7)
        
        # Add small circle at gauge end
        ax.scatter(gauge_x, gauge_y, s=20, color=color, zorder=5)
        
        # Add label
        ha = align if align != 'center' else 'center'
        ax.text(label_x, label_y, label, ha=ha, va='center', 
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor=color, alpha=0.9))
    
    # Scale markers (simplified)
    for i in range(0, 101, 20):
        angle = np.radians(180 - (i/100 * 180))
        outer_x = 1.05 * np.cos(angle)
        outer_y = 1.05 * np.sin(angle)
        inner_x = 0.95 * np.cos(angle)
        inner_y = 0.95 * np.sin(angle)
        
        ax.plot([inner_x, outer_x], [inner_y, outer_y], 
                color=colors['excellent'], linewidth=2, alpha=0.7)
        
        # Scale numbers (only major marks)
        if i % 20 == 0:
            text_x = 1.15 * np.cos(angle)
            text_y = 1.15 * np.sin(angle)
            ax.text(text_x, text_y, f'{i}', ha='center', va='center', 
                    fontsize=8, color=colors['excellent'], alpha=0.8,
                    fontweight='bold')
    
    # Redesigned metrics section - vertical layout to avoid overlap
    metrics_data = [
        ("Critical Events Detection", "100%", "●"),
        ("Alert Reduction", "98.8-99.4%", "▲"), 
        ("False Negative Rate", "0.0%", "■")
    ]
    
    box_width = 2.4
    box_height = 0.25
    start_y = -0.55
    
    # Single background box for all metrics
    metrics_box = FancyBboxPatch((-box_width/2, start_y - box_height/2), 
                                box_width, box_height, 
                                boxstyle="round,pad=0.03", 
                                facecolor='white', 
                                edgecolor=colors['good'],
                                linewidth=2,
                                alpha=0.95)
    ax.add_patch(metrics_box)
    
    # Display metrics in a single line with separators
    metrics_text = "● Critical Events: 100%   |   ▲ Alert Reduction: 98.8-99.4%   |   ■ False Negatives: 0.0%"
    
    ax.text(0, start_y, metrics_text, ha='center', va='center', 
            fontsize=9, fontweight='bold', color=colors['excellent'])
    
    # Add subtle highlight for perfect scores
    #ax.text(0, start_y - 0.08, "Perfect Safety Performance", ha='center', va='center', 
    #        fontsize=8, style='italic', color=colors['good'])
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    return fig, ax

# Create the professional gauge
fig, ax = create_professional_gauge()

plt.tight_layout()

# Save with high quality
plt.savefig('heliot_professional_safety_gauge_v2.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.1)
plt.savefig('heliot_professional_safety_gauge_v2.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.1)

plt.show()

print("✅ Improved gauge saved:")
print("🎯 heliot_professional_safety_gauge_v2.png / .pdf")