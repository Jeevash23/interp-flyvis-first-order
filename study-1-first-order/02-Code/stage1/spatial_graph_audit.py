"""Validate conservative type-graph zeros on the actual spatial graph, without interventions."""
from common import *
from core import *
torch.set_num_threads(2);admit();a=Core('000','cpu');source=a.source.numpy();target=a.target.numpy();reachable=np.zeros(a.n,dtype=bool);reachable[a.readout.numpy()]=True;rounds=[]
while True:
 new=reachable.copy();new[source[reachable[target]]]=True;rounds.append(int(new.sum()))
 if np.array_equal(new,reachable):break
 reachable=new
spatial_zero_types=[name for name in a.names if not reachable[a.node_names==name].any()];pair_has_path={pair:False for pair in a.pairs}
for s,t in zip(a.node_names[source[reachable[target]]],a.node_names[target[reachable[target]]]):pair_has_path[(s,t)]=True
spatial_zero_pairs=[list(pair) for pair,has_path in pair_has_path.items() if not has_path];type_zero_pairs=[list(pair) for pair in a.edge_zeros]
record={'status':'pass' if spatial_zero_types==a.zeros and spatial_zero_pairs==type_zero_pairs else 'additional_spatial_zeros_found','model':'000','structural_graph_only_no_stage2_interventions':True,'readouts':TARGETS,'reachable_spatial_nodes':int(reachable.sum()),'total_nodes':a.n,'backward_reachability_rounds':rounds,'structural_zero_types':spatial_zero_types,'structural_zero_type_pairs':spatial_zero_pairs,'realized_type_pairs':len(a.pairs),'type_graph_zero_count':len(type_zero_pairs),'all_main_interventions_still_retained':True};save_json(OUT/'verification/spatial-reachability.json',record);print(json.dumps(record,indent=2))
