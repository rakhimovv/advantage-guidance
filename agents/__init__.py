from agents.advantage_guidance import AdvantageGuidanceAgent
from agents.rcp import RCPAgent
from agents.bc import BCAgent
from agents.cfgrl import CFGRLAgent
from agents.dsrl import DSRLAgent
from agents.edp import EDPAgent
from agents.fawac import FAWACAgent
from agents.fbrac import FBRACAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.iql_diffusion import IQLDiffusionAgent
from agents.qam import QAMAgent
from agents.qgf import QGFAgent
from agents.sac import SACAgent

agents = dict(
    advantage_guidance=AdvantageGuidanceAgent,
    rcp=RCPAgent,
    bc=BCAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    iql_diffusion=IQLDiffusionAgent,
    cfgrl=CFGRLAgent,
    qgf=QGFAgent,
    sac=SACAgent,
    qam=QAMAgent,
    edp=EDPAgent,
    dsrl=DSRLAgent,
    fawac=FAWACAgent,
    fbrac=FBRACAgent,
)
