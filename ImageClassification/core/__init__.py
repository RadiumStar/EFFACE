from .federatedbase import Client, Server
from .fedavg import ClientFedAvg, ServerFedAvg

from .fedretrain import ClientFedRetrain, ServerFedRetrain

from .fedgb import ClientFedGB, ServerFedGB
from .fedgd import ClientFedGD, ServerFedGD 
from .fedga import ClientFedGA, ServerFedGA
from .fednot import ClientFedNot, ServerFedNot
from .fedmemprune import ClientFedMemPrune, ServerFedMemPrune
from .fedpsga import ClientFedPSGA, ServerFedPSGA
from .feduosc import ClientFedUOSC, ServerFedUOSC
from .fedrl import ClientFedRL, ServerFedRL             # Our base method
from .fedrl_gp import ClientFedRLGP, ServerFedRLGP      # Ours with LGP (Local Gradient Projection)
from .fedrl_gru import ClientFedRLGRU, ServerFedRLGRU   # Ours with GRU


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