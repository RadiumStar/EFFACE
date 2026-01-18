from .federatedbase import Client, Server
from .fedavg import ClientFedAvg, ServerFedAvg

from .fedretrain import ClientFedRetrain, ServerFedRetrain

from .fedgd import ClientFedGD, ServerFedGD 
from .fedga import ClientFedGA, ServerFedGA
from .fedrl import ClientFedRL, ServerFedRL


def get_client_class(method: str) -> type[Client]:
    """Get the client class based on the method name."""
    try: 
        return eval(f"Client{method}")
    except NameError:
        raise ValueError(f"Unknown client method: {method}")


def get_server_class(method: str) -> type[Server]:
    """Get the server class based on the method name."""
    try: 
        return eval(f"Server{method}")
    except NameError:
        raise ValueError(f"Unknown server method: {method}")