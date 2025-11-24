from alphanet.infer.calc import AlphaNetCalculator
#from alphanet.infer.new_haiku import AlphaNetCalculator #JAX version
from alphanet.config import All_Config
from ase.build import bulk
# example usage
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)*(3,3,3)*(3,3,3)
atoms.pbc = False
calculator = AlphaNetCalculator(
        ckpt_path='./pretrained/AQCAT25/aqcat_1021.ckpt',#./pretrained/OMA/haiku/haiku_params.pkl haiku ckpt
        device = 'cuda',
        precision = '32',
        config=All_Config().from_json('./pretrained/AQCAT25/aqcat.json'),
)

atoms.calc = calculator
print(atoms.get_potential_energy())
