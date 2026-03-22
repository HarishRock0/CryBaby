from airflow.models import DagBag
import os
import sys

# Add root and dags directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dags'))

def test_dag_integrity():
    """Test that the Airflow DAG can be loaded without errors."""
    dag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dags')
    dagbag = DagBag(dag_folder=dag_path, include_examples=False)
    
    # Check if there are any import errors
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
    
    # Check if our specific DAG is present
    dag_id = 'crybaby_training_pipeline'
    assert dag_id in dagbag.dags, f"DAG {dag_id} not found in DagBag"
