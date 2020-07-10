from util.plotting import coordinates_transform
import pickle

affine_trans_dict, srcTri_dict, dstTri_dict = (dict() for _ in range(3))

## GDGT tie Points
srcTri_dict['gdgt'] = [[65,96],[221,35],[299,39]]
dstTri_dict['gdgt'] = [[37,48],[352,769],[355,1138]]

## UK tie Points
srcTri_dict['uk'] = [[88,88], [235,25], [307,91]]
dstTri_dict['uk'] = [[37,90], [354,777], [30,1123]]

## Sterol tie Points
srcTri_dict['sterol'] = [[89,92],[244,22],[324,19]]
dstTri_dict['sterol'] = [[37,37],[360,759],[360,1132]]

## PAH tie points
srcTri_dict['pah'] = [[51,99],[205,37],[283,36]]
dstTri_dict['pah'] = [[37,37],[360,759],[360,1132]]

for data in ['gdgt','uk','sterol','pah']:
    affine_trans_dict[data] = dict()
    affine_trans_dict[data]['A'], affine_trans_dict[data]['t'] = coordinates_transform(srcTri_dict[data],dstTri_dict[data])

with open (r'./Dict/affine_trans_dict.pkl','wb') as f:
    pickle.dump(affine_trans_dict,f)