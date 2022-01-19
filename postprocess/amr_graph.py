import penman

def create_pediction_dict(X_test, y_pred, label_encoder):
    predictions = []
    cur_sentence_id = X_test['sentence_id'].values[0]
    i = 0

    while i < len(y_pred):
        prediction = {
            'nodes': [],
            'heads': [],
            'corefs': [],
            'head_labels': [],
            'sentence_id': 0
        }
        
        current_pairs = []
        root = None
        # Collect nodes
        while i < len(y_pred) and cur_sentence_id == X_test['sentence_id'].values[i]:
            # add nodes
            parent = X_test['parent'].values[i]
            child = X_test['child'].values[i]
            is_root = X_test['is_root'].values[i]
            
            if root is None and is_root==1:
                root = parent
            
            if (parent not in prediction['nodes']):
                prediction['nodes'].append(parent)
            if (child not in prediction['nodes']):
                prediction['nodes'].append(child)
            current_pairs.append((parent, child, y_pred[i]))    
            i+=1
        
        # Collect heads
        for node in prediction['nodes']:
            if node == root:
                prediction['heads'].append(0)
                prediction['head_labels'].append(':root')
            else:
                for pair in current_pairs:
                    if (pair[1] == node):
                        prediction['heads'].append(prediction['nodes'].index(pair[0])+1)
                        prediction['head_labels'] += list(label_encoder.inverse_transform([pair[2]]))

        prediction['corefs'] = list(range(1, len(prediction['nodes'])+1))
        prediction['sentence_id'] = cur_sentence_id
        prediction['root'] = root
        
        predictions.append(prediction)
        if (i < len(y_pred)):
            cur_sentence_id = X_test['sentence_id'].values[i]

    return predictions


def create_amr_graph_from_prediction(prediction):
    nodes = prediction['nodes']
    heads = prediction['heads']
    head_labels = prediction['head_labels']
    sentence_id = prediction['sentence_id']
    root = prediction['root']
    
    
    variable_map = dict()
    triples = []
    for idx, node in enumerate(nodes):
        variable_map['vv'+str(idx+1)] = node
    
    # find top 
    top = 'vv1'
    for var, value in variable_map.items():
        if value == root:
            top = var
    
    # rename nodes
    for key in variable_map:
        variable_map[key] = variable_map[key].split('_')[0]
    
    # create instances
    for variable in variable_map:
        triples.append((variable, 'instance', variable_map[variable]))

    # create connections
    for idx, head in enumerate(heads):
        if (head != 0):
            head_var = 'vv{}'.format(head)
            dep_var = 'vv{}'.format(idx+1)
            label = head_labels[idx]
            triple = (head_var, label, dep_var)
            triples.append(triple)
        
    graph = penman.Graph()
    
    graph.heads = heads
    graph.nodes = nodes
    graph.head_labels = head_labels
    graph._top = top
    graph._triples = [penman.Triple(*t) for t in triples]
    graph.id = sentence_id
    
    return graph

def save_amr_graphs(list_pair_sentence_amr, filepath):
    id_now = 1
    with open(filepath, 'w+', encoding='utf-8') as f:
        for sentence, amr_graph in list_pair_sentence_amr:
            try:               
                f.write("# ::id {}\n".format(id_now))
                id_now+=1

                f.write("# ::snt {}\n".format(sentence))
                f.write(str(amr_graph))
            except Exception as e:
                print(amr_graph._triples)
                print(sentence)
            f.write('\n\n')