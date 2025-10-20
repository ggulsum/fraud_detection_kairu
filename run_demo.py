#!/usr/bin/env python3
"""
Fraud Detection Demo Script
Tüm özellikleri göstermek için kapsamlı demo
"""

import os
import sys
import warnings
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from pipeline import FraudDetectionPipeline
from preprocessing import FeaturePreprocessor, ImbalanceHandler
from roc_auc import FraudEvaluator
from explainability import ModelExplainer
from outlier_detection import OutlierDetector

warnings.filterwarnings('ignore')


def demo_preprocessing():
    """Preprocessing demo"""
    print("\n" + "="*60)
    print("🔧 PREPROCESSING DEMO")
    print("="*60)
    
    # Import demo function
    from preprocessing import demo_preprocessing
    return demo_preprocessing()


def demo_outlier_detection():
    """Outlier detection demo"""
    print("\n" + "="*60)
    print("🎯 OUTLIER DETECTION DEMO")
    print("="*60)
    
    from outlier_detection import demo_with_synthetic_data
    return demo_with_synthetic_data()


def demo_evaluation():
    """Evaluation demo"""
    print("\n" + "="*60)
    print("📊 EVALUATION DEMO")
    print("="*60)
    
    from evaluation import demo_evaluation
    return demo_evaluation()


def demo_explainability():
    """Explainability demo"""
    print("\n" + "="*60)
    print("🔍 EXPLAINABILITY DEMO")
    print("="*60)
    
    if not EXPLAINABILITY_AVAILABLE:
        print("⚠️  SHAP/LIME explainability mevcut değil")
        print("💡 Sadece permutation importance ile devam ediliyor...")
        
        # Basic feature importance demo
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.inspection import permutation_importance
        import matplotlib.pyplot as plt
        
        X, y = make_classification(n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        # Plot
        feature_names = [f'feature_{i}' for i in range(10)]
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, perm_importance.importances_mean)
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance (Permutation-based)')
        plt.tight_layout()
        plt.show()
        
        print("✅ Basic explainability demo tamamlandı")
        return None
    
    return demo_explainability()


def demo_full_pipeline():
    """Full pipeline demo"""
    print("\n" + "="*60)
    print("🚀 FULL PIPELINE DEMO")
    print("="*60)
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline()
    
    print("Pipeline configuration loaded:")
    print(f"- Models: {list(pipeline.config.get('models', {}).keys())}")
    print(f"- Preprocessing: {pipeline.config.get('preprocessing', {}).get('scaling_method', 'default')}")
    print(f"- Evaluation thresholds: ROC-AUC >= {pipeline.config.get('evaluation', {}).get('min_roc_auc', 0.7)}")
    
    # Run full pipeline
    print("\n🏃‍♂️ Running full pipeline...")
    success = pipeline.run_full_pipeline(data_path=None, save_models=True)
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        
        # Show best model results
        best_model = pipeline._find_best_model()
        if best_model and best_model in pipeline.evaluators:
            results = pipeline.evaluators[best_model].results
            print(f"\n🏆 Best Model: {best_model}")
            print(f"   ROC-AUC: {results['roc_auc']:.4f}")
            print(f"   PR-AUC:  {results['pr_auc']:.4f}")
            print(f"   F1:      {results['f1_score']:.4f}")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall:    {results['recall']:.4f}")
        
        # Demo prediction
        print("\n🔮 Demo Predictions:")
        test_sample = pipeline.X_test.head(3)
        predictions, probabilities = pipeline.predict(test_sample, best_model)
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
            print(f"   Transaction {i+1}: {'FRAUD' if pred == 1 else 'NORMAL'} "
                  f"(Probability: {prob:.3f}, Risk: {risk_level})")
        
        return pipeline
    else:
        print("\n❌ Pipeline failed!")
        return None


def demo_model_comparison():
    """Model comparison demo"""
    print("\n" + "="*60)
    print("⚖️  MODEL COMPARISON DEMO")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_classification(
        n_samples=2000, n_features=15, n_informative=10,
        n_redundant=5, n_clusters_per_class=1,
        weights=[0.9, 0.1], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train and evaluate
    print("Training and evaluating models...")
    
    # Apply SMOTE for imbalance
    X_train_balanced, y_train_balanced = ImbalanceHandler.apply_smote(X_train, y_train)
    
    # Train models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_balanced, y_train_balanced)
    
    # Compare models
    evaluator = FraudEvaluator()
    comparison_df, comparison_results = evaluator.compare_models(models, X_test, y_test)
    
    print("\n📊 Model Comparison Results:")
    print(comparison_df.round(4))
    
    return comparison_df, comparison_results


def demo_business_metrics():
    """Business metrics demo"""
    print("\n" + "="*60)
    print("💰 BUSINESS METRICS DEMO")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Generate fraud dataset
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        weights=[0.95, 0.05], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Business cost analysis
    print("Business Impact Analysis:")
    print("-" * 30)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Cost parameters
    fraud_loss_per_case = 1000  # Average fraud loss
    investigation_cost = 50     # Cost to investigate each alert
    
    print(f"Assumptions:")
    print(f"- Average fraud loss: ${fraud_loss_per_case}")
    print(f"- Investigation cost per alert: ${investigation_cost}")
    print(f"- Total frauds in test set: {np.sum(y_test)}")
    print(f"- Total test transactions: {len(y_test)}")
    
    print(f"\nThreshold Analysis:")
    print("Threshold | Alerts | Caught | Missed | Total Cost | Cost per Transaction")
    print("-" * 75)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Confusion matrix
        tp = np.sum((y_test == 1) & (y_pred == 1))  # Frauds caught
        fp = np.sum((y_test == 0) & (y_pred == 1))  # False alarms
        fn = np.sum((y_test == 1) & (y_pred == 0))  # Frauds missed
        
        # Business metrics
        total_alerts = tp + fp
        frauds_caught = tp
        frauds_missed = fn
        
        # Costs
        investigation_costs = total_alerts * investigation_cost
        fraud_losses = frauds_missed * fraud_loss_per_case
        total_cost = investigation_costs + fraud_losses
        cost_per_transaction = total_cost / len(y_test)
        
        print(f"{threshold:8.1f} | {total_alerts:6d} | {frauds_caught:6d} | {frauds_missed:6d} | "
              f"${total_cost:8.0f} | ${cost_per_transaction:8.2f}")
    
    # Optimal threshold (minimum cost)
    costs = []
    for threshold in np.linspace(0.1, 0.9, 50):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        
        total_cost = (tp + fp) * investigation_cost + fn * fraud_loss_per_case
        costs.append(total_cost)
    
    optimal_threshold = np.linspace(0.1, 0.9, 50)[np.argmin(costs)]
    min_cost = min(costs)
    
    print(f"\n🎯 Optimal threshold: {optimal_threshold:.3f}")
    print(f"💰 Minimum total cost: ${min_cost:.0f}")
    print(f"💰 Cost per transaction: ${min_cost/len(y_test):.2f}")


def interactive_demo():
    """Interactive demo menu"""
    while True:
        print("\n" + "="*60)
        print("🎭 FRAUD DETECTION INTERACTIVE DEMO")
        print("="*60)
        print("1. 🔧 Preprocessing Demo")
        print("2. 🎯 Outlier Detection Demo")
        print("3. 📊 Evaluation Demo")
        print("4. 🔍 Explainability Demo")
        print("5. 🚀 Full Pipeline Demo")
        print("6. ⚖️  Model Comparison Demo")
        print("7. 💰 Business Metrics Demo")
        print("8. 🔄 Run All Demos")
        print("9. ❌ Exit")
        print("-" * 60)
        
        choice = input("Enter your choice (1-9): ").strip()
        
        try:
            if choice == '1':
                demo_preprocessing()
            elif choice == '2':
                demo_outlier_detection()
            elif choice == '3':
                demo_evaluation()
            elif choice == '4':
                demo_explainability()
            elif choice == '5':
                demo_full_pipeline()
            elif choice == '6':
                demo_model_comparison()
            elif choice == '7':
                demo_business_metrics()
            elif choice == '8':
                print("🔄 Running all demos...")
                demo_preprocessing()
                demo_outlier_detection()
                demo_evaluation()
                demo_explainability()
                demo_full_pipeline()
                demo_model_comparison()
                demo_business_metrics()
                print("\n✅ All demos completed!")
            elif choice == '9':
                print("👋 Demo finished. Thank you!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-9.")
        
        except KeyboardInterrupt:
            print("\n⏸️  Demo interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error in demo: {e}")
            print("Continuing to menu...")
        
        input("\nPress Enter to continue...")


def main():
    """Main function"""
    print("🎭 Welcome to Fraud Detection Demo!")
    print("This demo showcases all components of the fraud detection system.")
    
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        print("\n🔄 Running automated demo...")
        demo_full_pipeline()
    else:
        interactive_demo()


if __name__ == "__main__":
    main()