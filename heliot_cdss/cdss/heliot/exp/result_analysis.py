import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import argparse

all_timings = []

def clean_labels(text):
    """Clean and standardize labels"""
    if pd.isna(text):
        return "UNKNOWN"
    return text.strip().upper()

def clean_reaction_types(row):
    """Clean and standardize reaction types"""
    if row['classification_descr'] == "DRUG CLASS CROSS-REACTIVITY WITH DOCUMENTED TOLERANCE":
        return "None"
    
    reaction = row['reaction_types']
    if pd.isna(reaction):
        return "None"
    return str(reaction).strip().upper()

def create_alert_type(reaction):
    """Create alert type based on reaction"""
    if pd.isna(reaction):
        return "NO ALERT NEEDED"
    
    reaction_str = str(reaction).strip()
    
    if reaction_str == "NON LIFE-THREATENING NON IMMUNE-MEDIATED":
        return "NON-INTERRUPTIVE ALERT"
    elif reaction_str in ["NON LIFE-THREATENING IMMUNE-MEDIATED", "LIFE-THREATENING"]:
        return "INTERRUPTIVE ALERT"
    else:
        return "NO ALERT NEEDED"

def create_confusion_matrix_class_plot(y_true, y_pred, labels, output_file):
    """Create and save the confusion matrix plot"""
    # Convert labels to all lower-case and first letter in upper-case
    formatted_labels = [label.capitalize() for label in labels]
    
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Use labels to format display 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=formatted_labels, yticklabels=formatted_labels)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Oblique x labels
    plt.xticks(rotation=45, ha='right')
    # Horizontal y labels
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def calculate_class_metrics(y_true, y_pred, labels):
    """Calculate per-class metrics"""
    class_metrics = {}
    for label in labels:
        y_true_binary = (y_true == label)
        y_pred_binary = (y_pred == label)
        
        class_metrics[label] = {
            'Precision': precision_score(y_true_binary, y_pred_binary),
            'Recall': recall_score(y_true_binary, y_pred_binary),
            'F1': f1_score(y_true_binary, y_pred_binary)
        }
    return class_metrics

### ALERT SAFETY FUNCTION ###
def analyze_safety_critical_cases(df_clean):
    """Analyze safety-critical false negatives and missed reactions"""
    
    # Identify false negatives for alerts (missed critical reactions)
    fn_alerts = df_clean[
        (df_clean['alert_true'].isin(['INTERRUPTIVE ALERT', 'NON-INTERRUPTIVE ALERT'])) & 
        (df_clean['alert_resp'] == 'NO ALERT NEEDED')
    ].copy()
    
    # Categorize by severity
    critical_missed = fn_alerts[fn_alerts['alert_true'] == 'INTERRUPTIVE ALERT']
    moderate_missed = fn_alerts[fn_alerts['alert_true'] == 'NON-INTERRUPTIVE ALERT']
    
    # Analyze confidence scores if available
    safety_analysis = {
        'total_false_negatives': len(fn_alerts),
        'critical_missed_count': len(critical_missed),
        'moderate_missed_count': len(moderate_missed),
        'critical_missed_percentage': (len(critical_missed) / len(df_clean)) * 100,
        'moderate_missed_percentage': (len(moderate_missed) / len(df_clean)) * 100,
        'safety_critical_cases': fn_alerts
    }
    
    return safety_analysis

def create_safety_focused_confusion_matrix(y_true, y_pred, labels, output_file, highlight_fn=True):
    """Create confusion matrix with safety focus highlighting false negatives"""
    
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create custom colormap to highlight false negatives
    if highlight_fn:
        # Create annotation matrix with special formatting for FN
        annot_matrix = []
        for i in range(len(labels)):
            row = []
            for j in range(len(labels)):
                if i != j and j == 0:  # False negatives (should be alert but predicted as no alert)
                    row.append(f"⚠️ {cm[i,j]}")
                else:
                    row.append(str(cm[i,j]))
            annot_matrix.append(row)
    else:
        annot_matrix = cm
    
    # Use custom color scheme
    cmap = plt.cm.Reds if highlight_fn else plt.cm.Blues
    
    formatted_labels = [label.replace('_', ' ').title() for label in labels]
    
    sns.heatmap(cm, annot=annot_matrix, fmt='', cmap=cmap, 
                xticklabels=formatted_labels, yticklabels=formatted_labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Safety-Focused Alert Confusion Matrix\n(⚠️ indicates safety-critical false negatives)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_clinical_safety_metrics(df_clean):
    """Calculate standard clinical safety metrics"""
    
    # Basic alert distribution (keep these as you said)
    total_cases = len(df_clean)
    no_alert_cases = len(df_clean[df_clean['alert_resp'] == 'NO ALERT NEEDED'])
    non_inter_alert_cases = len(df_clean[df_clean['alert_resp'] == 'NON-INTERRUPTIVE ALERT'])
    inter_alert_cases = len(df_clean[df_clean['alert_resp'] == 'INTERRUPTIVE ALERT'])
    
    alert_reduction_percentage = (no_alert_cases / total_cases) * 100
    non_interruptive_alert_percentage = (non_inter_alert_cases/ total_cases) * 100
    interruptive_alert_percentage = (inter_alert_cases/ total_cases) * 100
    
    # Standard clinical safety metrics
    # For CRITICAL (life-threatening) reactions
    critical_tp = len(df_clean[
        (df_clean['alert_true'] == 'INTERRUPTIVE ALERT') & 
        (df_clean['alert_resp'] == 'INTERRUPTIVE ALERT')
    ])
    
    critical_fn = len(df_clean[
        (df_clean['alert_true'] == 'INTERRUPTIVE ALERT') & 
        (df_clean['alert_resp'] != 'INTERRUPTIVE ALERT')  # Any non-interruptive response
    ])
    
    critical_fp = len(df_clean[
        (df_clean['alert_true'] != 'INTERRUPTIVE ALERT') & 
        (df_clean['alert_resp'] == 'INTERRUPTIVE ALERT')
    ])
    
    critical_tn = len(df_clean[
        (df_clean['alert_true'] != 'INTERRUPTIVE ALERT') & 
        (df_clean['alert_resp'] != 'INTERRUPTIVE ALERT')
    ])
    
    # For ANY alert needed vs no alert
    any_alert_tp = len(df_clean[
        (df_clean['alert_true'] != 'NO ALERT NEEDED') & 
        (df_clean['alert_resp'] != 'NO ALERT NEEDED')
    ])
    
    any_alert_fn = len(df_clean[
        (df_clean['alert_true'] != 'NO ALERT NEEDED') & 
        (df_clean['alert_resp'] == 'NO ALERT NEEDED')
    ])
    
    any_alert_fp = len(df_clean[
        (df_clean['alert_true'] == 'NO ALERT NEEDED') & 
        (df_clean['alert_resp'] != 'NO ALERT NEEDED')
    ])
    
    any_alert_tn = len(df_clean[
        (df_clean['alert_true'] == 'NO ALERT NEEDED') & 
        (df_clean['alert_resp'] == 'NO ALERT NEEDED')
    ])
    
    # Calculate standard metrics
    # Critical reactions metrics
    critical_sensitivity = critical_tp / (critical_tp + critical_fn) if (critical_tp + critical_fn) > 0 else 0
    critical_specificity = critical_tn / (critical_tn + critical_fp) if (critical_tn + critical_fp) > 0 else 0
    critical_ppv = critical_tp / (critical_tp + critical_fp) if (critical_tp + critical_fp) > 0 else 0
    critical_npv = critical_tn / (critical_tn + critical_fn) if (critical_tn + critical_fn) > 0 else 0
    
    # Any alert metrics
    any_alert_sensitivity = any_alert_tp / (any_alert_tp + any_alert_fn) if (any_alert_tp + any_alert_fn) > 0 else 0
    any_alert_specificity = any_alert_tn / (any_alert_tn + any_alert_fp) if (any_alert_tn + any_alert_fp) > 0 else 0
    any_alert_ppv = any_alert_tp / (any_alert_tp + any_alert_fp) if (any_alert_tp + any_alert_fp) > 0 else 0
    any_alert_npv = any_alert_tn / (any_alert_tn + any_alert_fn) if (any_alert_tn + any_alert_fn) > 0 else 0
    
    # Clinical risk metrics
    missed_critical_rate = critical_fn / total_cases * 100  # % of all cases that are missed critical
    false_alarm_rate = (critical_fp + any_alert_fp) / total_cases * 100  # % of all cases that are false alarms
    
    # Alert burden reduction
    original_alert_burden = len(df_clean[df_clean['alert_true'] != 'NO ALERT NEEDED'])  # How many would have had alerts originally
    current_alert_burden = len(df_clean[df_clean['alert_resp'] != 'NO ALERT NEEDED'])   # How many have alerts now
    alert_burden_reduction = ((original_alert_burden - current_alert_burden) / original_alert_burden * 100) if original_alert_burden > 0 else 0
    
    print(f"no_alert_cases: {no_alert_cases} non_inter_alert_cases: {non_inter_alert_cases} total_cases: {total_cases}")
    alert_reduction_percentage = (no_alert_cases - 41 + non_inter_alert_cases)/total_cases # The total number of no_alert_cases for trad. CDSS is 41

    return {
        # Basic distribution (keep as requested)
        'no_alert_cases': no_alert_cases,
        'no_alert_needed_percentage': alert_reduction_percentage,
        'non_inter_alert_cases': non_inter_alert_cases,
        'non_interruptive_alert_percentage': non_interruptive_alert_percentage,
        'inter_alert_cases': inter_alert_cases,
        'interruptive_alert_percentage': interruptive_alert_percentage,
        'alert_reduction_percentage': alert_reduction_percentage,
        
        # Standard clinical metrics for CRITICAL reactions
        'critical_sensitivity': critical_sensitivity,  # Recall for life-threatening
        'critical_specificity': critical_specificity,  # True negative rate for life-threatening
        'critical_ppv': critical_ppv,                  # Positive predictive value
        'critical_npv': critical_npv,                  # Negative predictive value
        'critical_false_negatives': critical_fn,       # Number of missed critical cases
        'critical_false_positives': critical_fp,       # Number of false critical alerts
        
        # Standard clinical metrics for ANY alert
        'any_alert_sensitivity': any_alert_sensitivity,
        'any_alert_specificity': any_alert_specificity,
        'any_alert_ppv': any_alert_ppv,
        'any_alert_npv': any_alert_npv,
        'any_alert_false_negatives': any_alert_fn,
        'any_alert_false_positives': any_alert_fp,
        
        # Clinical risk assessment
        'missed_critical_rate_percent': missed_critical_rate,
        'false_alarm_rate_percent': false_alarm_rate,
        'alert_burden_reduction_percent': alert_burden_reduction,
        
        # Confusion matrix components for reporting
        'critical_confusion_matrix': {
            'tp': critical_tp, 'fp': critical_fp,
            'fn': critical_fn, 'tn': critical_tn
        },
        'any_alert_confusion_matrix': {
            'tp': any_alert_tp, 'fp': any_alert_fp,
            'fn': any_alert_fn, 'tn': any_alert_tn
        }
    }

def create_clinical_safety_dashboard(df_clean, clinical_metrics, output_file):
    """Create comprehensive clinical safety visualization dashboard"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Alert distribution pie chart
    alert_counts = df_clean['alert_resp'].value_counts()
    colors = ['#e74c3c' if 'NO ALERT' in x else '#f39c12' if 'NON-INTERRUPTIVE' in x else '#3498db' 
              for x in alert_counts.index]
    
    wedges, texts, autotexts = ax1.pie(alert_counts.values, 
                                      labels=[x.replace('_', ' ').replace('ALERT', '').strip().title() 
                                             for x in alert_counts.index], 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Alert Response Distribution', fontsize=12, fontweight='bold', pad=20)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 2. Confusion matrix heatmap (more clinical focus)
    safety_outcomes = pd.crosstab(df_clean['alert_true'], df_clean['alert_resp'], margins=False)
    
    # Reorder for clinical interpretation (most critical first)
    desired_order = ['INTERRUPTIVE ALERT', 'NON-INTERRUPTIVE ALERT', 'NO ALERT NEEDED']
    safety_outcomes = safety_outcomes.reindex(index=desired_order, columns=desired_order, fill_value=0)
    
    sns.heatmap(safety_outcomes, annot=True, fmt='d', cmap='RdYlBu_r', ax=ax2,
                cbar_kws={'label': 'Number of Cases'})
    ax2.set_title('Alert Decision Matrix', fontsize=12, fontweight='bold', pad=20)
    ax2.set_xlabel('Predicted Alert Type')
    ax2.set_ylabel('Required Alert Type')
    
    # Highlight dangerous cells (false negatives)
    for i in range(len(desired_order)):
        for j in range(len(desired_order)):
            if i < j:  # Upper triangle = missed more severe alerts
                ax2.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
    
    # 3. Clinical risk categorization
    risk_categories = []
    
    for _, row in df_clean.iterrows():
        if row['alert_true'] == 'INTERRUPTIVE ALERT' and row['alert_resp'] != 'INTERRUPTIVE ALERT':
            risk_categories.append('Critical Miss')
        elif row['alert_true'] == 'NON-INTERRUPTIVE ALERT' and row['alert_resp'] == 'NO ALERT NEEDED':
            risk_categories.append('Moderate Miss')
        elif row['alert_true'] == 'NO ALERT NEEDED' and row['alert_resp'] != 'NO ALERT NEEDED':
            risk_categories.append('False Alarm')
        else:
            risk_categories.append('Correct Decision')
    
    risk_counts = pd.Series(risk_categories).value_counts()
    risk_colors = {
        'Critical Miss': '#e74c3c',      # Red
        'Moderate Miss': '#f39c12',      # Orange  
        'False Alarm': '#f1c40f',        # Yellow
        'Correct Decision': '#27ae60'    # Green
    }
    
    bars = ax3.bar(risk_counts.index, risk_counts.values, 
                   color=[risk_colors.get(x, '#95a5a6') for x in risk_counts.index])
    ax3.set_title('Clinical Risk Assessment', fontsize=12, fontweight='bold', pad=20)
    ax3.set_ylabel('Number of Cases')
    ax3.tick_params(axis='x', rotation=45)
    
    # FIX 1: Better positioning for percentage labels
    max_height = max(risk_counts.values)
    total_cases = len(df_clean)
    
    for bar, count in zip(bars, risk_counts.values):
        height = bar.get_height()
        percentage = (count / total_cases) * 100
        
        # Position label with proper offset based on bar height
        label_y = height + max_height * 0.05  # 5% offset from max height
        
        ax3.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{int(count)}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Set y-axis limit to accommodate labels
    ax3.set_ylim(0, max_height * 1.25)
    
    # 4. Performance metrics comparison
    metrics_data = {
        'Critical Alerts': [
            clinical_metrics['critical_sensitivity'],
            clinical_metrics['critical_specificity'],
            clinical_metrics['critical_ppv'],
            clinical_metrics['critical_npv']
        ],
        'Any Alert': [
            clinical_metrics['any_alert_sensitivity'],
            clinical_metrics['any_alert_specificity'], 
            clinical_metrics['any_alert_ppv'],
            clinical_metrics['any_alert_npv']
        ]
    }
    
    metric_names = ['Sensitivity\n(Recall)', 'Specificity', 'PPV\n(Precision)', 'NPV']
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, metrics_data['Critical Alerts'], width, 
                    label='Critical Alerts', color='#e74c3c', alpha=0.8)
    bars2 = ax4.bar(x + width/2, metrics_data['Any Alert'], width,
                    label='Any Alert', color='#3498db', alpha=0.8)
    
    ax4.set_ylabel('Score')
    ax4.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold', pad=20)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names)
    
    # FIX 2: Position legend outside plot area
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax4.set_ylim(0, 1.15)  # More space at top for value labels
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars with better spacing
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if there's a value
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Adjust spacing between subplots with more room for legend
    plt.tight_layout(pad=3.0, rect=[0, 0, 0.95, 1])  # Leave space on right for legend
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_false_negative_analysis_plot(df_clean, output_file):
    """Create focused analysis of false negatives (missed alerts) comparing HELIOT vs Traditional CDSS"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. False Negative Breakdown (for HELIOT)
    fn_data = []
    fn_labels = []
    
    # Critical false negatives (most dangerous)
    critical_fn = len(df_clean[
        (df_clean['alert_true'] == 'INTERRUPTIVE ALERT') & 
        (df_clean['alert_resp'] != 'INTERRUPTIVE ALERT')
    ])
    
    moderate_fn = len(df_clean[
        (df_clean['alert_true'] == 'NON-INTERRUPTIVE ALERT') & 
        (df_clean['alert_resp'] == 'NO ALERT NEEDED')
    ])
    
    fn_data = [critical_fn, moderate_fn]
    fn_labels = ['Missed Critical\n(Life-threatening)', 'Missed Moderate\n(Non-critical)']
    colors = ['#e74c3c', '#f39c12']
    
    bars = ax1.bar(fn_labels, fn_data, color=colors, alpha=0.8)
    ax1.set_title('HELIOT: False Negatives Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Missed Cases')
    
    # Add value and percentage labels
    total_cases = len(df_clean)
    for bar, count in zip(bars, fn_data):
        height = bar.get_height()
        percentage = (count / total_cases) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(count)}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Alert Burden: HELIOT vs Traditional CDSS
    # Traditional CDSS: triggers alert unless classification is one of the "safe" categories
    safe_classifications = [
        'NO REACTIVITY TO PRESCRIBED DRUG\'S INGREDIENTS OR EXCIPIENTS',
        'NO DOCUMENTED REACTIONS OR INTOLERANCES'
    ]
    
    # Traditional CDSS behavior
    traditional_no_alert = len(df_clean[
        df_clean['classification_descr'].isin(safe_classifications)
    ])
    traditional_alert = total_cases - traditional_no_alert
    
    # HELIOT behavior
    heliot_no_alert = len(df_clean[df_clean['alert_resp'] == 'NO ALERT NEEDED'])
    heliot_alert = total_cases - heliot_no_alert
    
    print(f"Traditional CDSS - Alerts: {traditional_alert}, No Alerts: {traditional_no_alert}")
    print(f"HELIOT - Alerts: {heliot_alert}, No Alerts: {heliot_no_alert}")
    
    categories = ['Alerts Generated', 'No Alert Cases']
    traditional_counts = [traditional_alert, traditional_no_alert]
    heliot_counts = [heliot_alert, heliot_no_alert]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, traditional_counts, width, label='Traditional CDSS', 
                    color='#95a5a6', alpha=0.7)
    bars2 = ax2.bar(x + width/2, heliot_counts, width, label='HELIOT (AI System)',
                    color='#3498db', alpha=0.8)
    
    ax2.set_ylabel('Number of Cases')
    ax2.set_title('Alert Burden: Traditional CDSS vs HELIOT', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Calculate and display alert reduction
    alert_reduction_count = traditional_alert - heliot_alert
    reduction_pct = (alert_reduction_count / traditional_alert * 100) if traditional_alert > 0 else 0
    
    ax2.text(0.5, 0.95, f'Alert Reduction: {alert_reduction_count} cases ({reduction_pct:.1f}%)', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

#############################


def main(iter, suffix):
    print("Loading results...")
    print(f'./results/{iter}/results_full_synth.xlsx')
    try:
        df = pd.read_excel(f'./results/{iter}/results_full_synth.xlsx', dtype={'drug_code': str, 'leaflet': str, 'patient_id': str, 'classification':str})  
    except Exception as e:
        print(f"Error while loading the results: {str(e)}")
        return

    print("\nPreparing data...")
    all_timings.extend(df['timing'].dropna().tolist())

    df['true_labels'] = df['classification_descr'].apply(clean_labels)
    df['pred_labels'] = df['classification_resp'].apply(clean_labels)
    
    df['true_reactions'] = df.apply(clean_reaction_types, axis=1)
    df['pred_reactions'] = df['reaction_resp'].apply(lambda x: "None" if pd.isna(x) else str(x).strip().upper())

    # Add alert columns
    df['alert_true'] = df['true_reactions'].apply(create_alert_type)
    df['alert_resp'] = df['pred_reactions'].apply(create_alert_type)

    df_clean = df.dropna(subset=['true_labels', 'pred_labels'])
    
    unique_labels = sorted(list(set(df_clean['true_labels'].unique()) | 
                              set(df_clean['pred_labels'].unique())))
    unique_reactions = sorted(list(set(df_clean['true_reactions'].unique()) | 
                                 set(df_clean['pred_reactions'].unique())))
    unique_alerts = sorted(list(set(df_clean['alert_true'].unique()) | 
                              set(df_clean['alert_resp'].unique())))

    present_labels = [label for label in unique_labels 
                    if sum(df_clean['true_labels'] == label) > 0]
    
    present_reactions = [label for label in unique_reactions 
                    if sum(df_clean['true_reactions'] == label) > 0]


    present_alerts = [label for label in unique_alerts 
                    if sum(df_clean['alert_true'] == label) > 0]

    print("\nCalculating classification metrics...")
    classification_metrics = {
        'Accuracy': accuracy_score(df_clean['true_labels'], df_clean['pred_labels']),
        'Macro Precision': precision_score(df_clean['true_labels'], 
                                         df_clean['pred_labels'], 
                                         average='macro',
                                         zero_division=1,
                                         labels=present_labels),
        'Macro Recall': recall_score(df_clean['true_labels'], 
                                   df_clean['pred_labels'], 
                                   average='macro',
                                   labels=present_labels),
        'Macro F1': f1_score(df_clean['true_labels'], 
                            df_clean['pred_labels'], 
                            average='macro',
                            labels=present_labels)
    }

    print("\nCalculating reaction metrics...")
    reaction_metrics = {
        'Accuracy': accuracy_score(df_clean['true_reactions'], df_clean['pred_reactions']),
        'Macro Precision': precision_score(df_clean['true_reactions'], 
                                         df_clean['pred_reactions'], 
                                         average='macro',
                                         labels=present_reactions),
        'Macro Recall': recall_score(df_clean['true_reactions'], 
                                   df_clean['pred_reactions'], 
                                   average='macro',
                                   labels=present_reactions),
        'Macro F1': f1_score(df_clean['true_reactions'], 
                            df_clean['pred_reactions'], 
                            average='macro',
                            labels=present_reactions)
    }

    print("\nCalculating alert metrics...")
    alert_metrics = {
        'Accuracy': accuracy_score(df_clean['alert_true'], df_clean['alert_resp']),
        'Macro Precision': precision_score(df_clean['alert_true'], 
                                         df_clean['alert_resp'], 
                                         average='macro',
                                         labels=present_alerts),
        'Macro Recall': recall_score(df_clean['alert_true'], 
                                   df_clean['alert_resp'], 
                                   average='macro',
                                   labels=present_alerts),
        'Macro F1': f1_score(df_clean['alert_true'], 
                            df_clean['alert_resp'], 
                            average='macro',
                            labels=present_alerts)
    }

    # Calculate per-class metrics
    classification_class_metrics = calculate_class_metrics(df_clean['true_labels'], 
                                                         df_clean['pred_labels'], 
                                                         unique_labels)
    
    reaction_class_metrics = calculate_class_metrics(df_clean['true_reactions'], 
                                                   df_clean['pred_reactions'], 
                                                   unique_reactions)
    
    alert_class_metrics = calculate_class_metrics(df_clean['alert_true'], 
                                                 df_clean['alert_resp'], 
                                                 unique_alerts)

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f'./results/{iter}/evaluation_report_{suffix}.txt'
    class_cm_filename = f'./results/{iter}/classification_confusion_matrix_{suffix}.png'
    reaction_cm_filename = f'./results/{iter}/reaction_confusion_matrix_{suffix}.png'
    alert_cm_filename = f'./results/{iter}/alert_confusion_matrix_{suffix}.png'

    create_confusion_matrix_class_plot(df_clean['true_labels'], 
                               df_clean['pred_labels'],
                               unique_labels,
                               class_cm_filename)
    
    create_confusion_matrix_class_plot(df_clean['true_reactions'], 
                               df_clean['pred_reactions'],
                               unique_reactions,
                               reaction_cm_filename)
    
    create_confusion_matrix_class_plot(df_clean['alert_true'], 
                               df_clean['alert_resp'],
                               unique_alerts,
                               alert_cm_filename)

    with open(report_filename, 'w') as f:
        f.write("EVALUATION REPORT\n")
        f.write("================\n\n")
        
        # Classification Metrics
        f.write("CLASSIFICATION METRICS\n")
        f.write("--------------------\n")
        f.write("Overall Metrics:\n")
        for metric, value in classification_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nPer-Class Metrics:\n")
        for label in unique_labels:
            f.write(f"\n{label}:\n")
            for metric, value in classification_class_metrics[label].items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        # Reaction Metrics
        f.write("\nREACTION METRICS\n")
        f.write("---------------\n")
        f.write("Overall Metrics:\n")
        for metric, value in reaction_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nPer-Class Metrics:\n")
        for reaction in unique_reactions:
            f.write(f"\n{reaction}:\n")
            for metric, value in reaction_class_metrics[reaction].items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        # Alert Metrics
        f.write("\nALERT METRICS\n")
        f.write("------------\n")
        f.write("Overall Metrics:\n")
        for metric, value in alert_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nPer-Class Metrics:\n")
        for alert in unique_alerts:
            f.write(f"\n{alert}:\n")
            for metric, value in alert_class_metrics[alert].items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        # Dataset Statistics
        f.write("\nDATASET STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Valid samples: {len(df_clean)}\n")
        f.write(f"Missing labels: {len(df) - len(df_clean)}\n")
        
        f.write("\nClassification distribution:\n")
        for label in unique_labels:
            true_count = sum(df_clean['true_labels'] == label)
            pred_count = sum(df_clean['pred_labels'] == label)
            correct_count = sum((df_clean['true_labels'] == label) & (df_clean['pred_labels'] == label))
            
            f.write(f"{label}:\n")
            f.write(f"  True labels: {true_count}\n")
            f.write(f"  Predicted labels: {pred_count}\n")
            f.write(f"  Correctly classified: {correct_count}\n")
            f.write(f"  Incorrectly classified: {true_count - correct_count}\n")
            
        f.write("\nReaction distribution:\n")
        for reaction in unique_reactions:
            true_count = sum(df_clean['true_reactions'] == reaction)
            pred_count = sum(df_clean['pred_reactions'] == reaction)
            correct_count = sum((df_clean['true_reactions'] == reaction) & (df_clean['pred_reactions'] == reaction))
            
            f.write(f"{reaction}:\n")
            f.write(f"  True reactions: {true_count}\n")
            f.write(f"  Predicted reactions: {pred_count}\n")
            f.write(f"  Correctly classified: {correct_count}\n")
            f.write(f"  Incorrectly classified: {true_count - correct_count}\n")
        
        f.write("\nAlert distribution:\n")
        for alert in unique_alerts:
            true_count = sum(df_clean['alert_true'] == alert)
            pred_count = sum(df_clean['alert_resp'] == alert)
            correct_count = sum((df_clean['alert_true'] == alert) & (df_clean['alert_resp'] == alert))
            
            f.write(f"{alert}:\n")
            f.write(f"  True alerts: {true_count}\n")
            f.write(f"  Predicted alerts: {pred_count}\n")
            f.write(f"  Correctly classified: {correct_count}\n")
            f.write(f"  Incorrectly classified: {true_count - correct_count}\n")



    ####### SAFETY ALERT #####
    # SAFETY ANALYSIS - Add this section in main() function
    # Add clinical safety analysis to the report
    clinical_metrics = calculate_clinical_safety_metrics(df_clean)

    # Create improved visualizations
    safety_dashboard_filename = f'./results/{iter}/clinical_safety_dashboard_{suffix}.png'
    false_negative_filename = f'./results/{iter}/false_negative_analysis_{suffix}.png'
    
    create_clinical_safety_dashboard(df_clean, clinical_metrics, safety_dashboard_filename)
    create_false_negative_analysis_plot(df_clean, false_negative_filename)

    
    with open(report_filename, 'a') as f:
        f.write("\n\nCLINICAL SAFETY ANALYSIS\n")
        f.write("=======================\n\n")
        
        f.write("ALERT DISTRIBUTION\n")
        f.write("-----------------\n")
        f.write(f"No Alert needed: {clinical_metrics['no_alert_cases']} ({clinical_metrics['no_alert_needed_percentage']*100:.1f}%)\n")
        f.write(f"Non-interruptive Alert: {clinical_metrics['non_inter_alert_cases']} ({clinical_metrics['non_interruptive_alert_percentage']:.1f}%)\n")
        f.write(f"Interruptive Alert: {clinical_metrics['inter_alert_cases']} ({clinical_metrics['interruptive_alert_percentage']:.1f}%)\n\n")
        f.write(f"Alert perc. Reduction vs Traditional CDSS: {clinical_metrics['alert_reduction_percentage']*100:.1f}%\n\n")  
        
        f.write("CRITICAL (LIFE-THREATENING) REACTION DETECTION\n")
        f.write("---------------------------------------------\n")
        f.write(f"Sensitivity (Recall): {clinical_metrics['critical_sensitivity']:.4f}\n")
        f.write(f"Specificity: {clinical_metrics['critical_specificity']:.4f}\n")
        f.write(f"Positive Predictive Value (Precision): {clinical_metrics['critical_ppv']:.4f}\n")
        f.write(f"Negative Predictive Value: {clinical_metrics['critical_npv']:.4f}\n")
        f.write(f"False Negatives (Missed Critical): {clinical_metrics['critical_false_negatives']}\n")
        f.write(f"False Positives (Unnecessary Critical Alerts): {clinical_metrics['critical_false_positives']}\n\n")
        
        f.write("ANY ALERT DETECTION\n")
        f.write("------------------\n")
        f.write(f"Sensitivity: {clinical_metrics['any_alert_sensitivity']:.4f}\n")
        f.write(f"Specificity: {clinical_metrics['any_alert_specificity']:.4f}\n")
        f.write(f"Positive Predictive Value: {clinical_metrics['any_alert_ppv']:.4f}\n")
        f.write(f"Negative Predictive Value: {clinical_metrics['any_alert_npv']:.4f}\n\n")
        
        f.write("CLINICAL RISK ASSESSMENT\n")
        f.write("-----------------------\n")
        f.write(f"Missed Critical Reaction Rate: {clinical_metrics['missed_critical_rate_percent']:.2f}%\n")
        f.write(f"False Alarm Rate: {clinical_metrics['false_alarm_rate_percent']:.2f}%\n")
        f.write(f"Alert Burden Reduction: {clinical_metrics['alert_burden_reduction_percent']:.2f}%\n\n")

        ##########################

        print("\nSaved Results:")
        print(f"- Report: {report_filename}")
        print(f"- Classification Confusion Matrix: {class_cm_filename}")
        print(f"- Reaction Confusion Matrix: {reaction_cm_filename}")
        print(f"- Alert Confusion Matrix: {alert_cm_filename}")
        
        print("\nClassification metrics summary:")
        for metric, value in classification_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nReaction metrics summary:")
        for metric, value in reaction_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nAlert metrics summary:")
        for metric, value in alert_metrics.items():
            print(f"{metric}: {value:.4f}")

    ##########################

    print("\nSaved Results:")
    print(f"- Report: {report_filename}")
    print(f"- Classification Confusion Matrix: {class_cm_filename}")
    print(f"- Reaction Confusion Matrix: {reaction_cm_filename}")
    print(f"- Alert Confusion Matrix: {alert_cm_filename}")
    
    print("\nClassification metrics summary:")
    for metric, value in classification_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nReaction metrics summary:")
    for metric, value in reaction_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nAlert metrics summary:")
    for metric, value in alert_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the file.")
    
    # Add arguments for the files
    parser.add_argument("--input", help="Path to the file input path.", default="gpt")

    # Parse the arguments
    args = parser.parse_args()

    main(args.input+'/I','I')
    main(args.input+'/II','II')
    main(args.input+'/III','III')
    main(args.input+'/IV','IV')
    main(args.input+'/V','V')
    
    average_timing = sum(all_timings) / len(all_timings)
    print(f"TOTAL AVERAGE 'timing': {average_timing}")