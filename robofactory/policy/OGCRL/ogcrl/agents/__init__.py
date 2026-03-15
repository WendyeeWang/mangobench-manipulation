from ogcrl.agents.crl import CRLAgent
from ogcrl.agents.gcbc import GCBCAgent
from ogcrl.agents.gciql import GCIQLAgent
from ogcrl.agents.gcivl import GCIVLAgent
from ogcrl.agents.hiql import HIQLAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
   
)
