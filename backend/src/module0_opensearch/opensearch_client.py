import os
import logging
from opensearchpy import OpenSearch

logger = logging.getLogger(__name__)

def get_opensearch_client():
    """
    Initializes and returns an OpenSearch client connection.
    Connects to a local or containerized OpenSearch instance (default: localhost:9200).
    """
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    
    auth = (os.getenv("OPENSEARCH_USER", "admin"), os.getenv("OPENSEARCH_PASSWORD", "admin"))
    
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=auth,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )
    
    try:
        if client.ping():
            logger.info("Successfully connected to OpenSearch cluster.")
        else:
            logger.warning("OpenSearch ping failed, cluster might be offline.")
    except Exception as e:
        logger.error(f"Error connecting to OpenSearch: {e}")
        
    return client