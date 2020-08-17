import json
import os
import argparse

from knowledge_bert.tokenization import BertTokenizer

def load_wikidata(prep_input_path: str):
  """Parses the preprocessed wikdata file into Python dicts.

  Args:
    prep_input_path: path to the preprocessed wikidata file

  Returns:
    wiki_title2id: dict, key is wiki_title and value is kg info.
      Currently not used for memory efficiency
    wiki_title2id: dict, key is wiki_title.lower and value is kg info.
      Uncased version is for second matching.
    entity_id2label: dict, key is entity_id and value is word form
  """
  wiki_title2id = {}
  # used for converting parent_id to word form if necessary.
  entity_id2label = {}
  entity_id2parents = {}
  
  with open(prep_input_path, "r") as fh:
    for idx, line in enumerate(fh):
      line = line.strip()
      # Skip empty lines
      if not line:
        continue
      items = line.split("\t")

      wiki_title = items[2]
      entity_id = items[0]
      # Dirty cases:  a few entity ids doesn't start with 'Q' and are discarded
      if entity_id[0] != "Q":
        continue

      # E.g., Q100->100 for converting into integers.
      # Converting into integers helps save much memory
      #  entity_id = int(entity_id[1:])
      # entity_label. E.g., George Washington is a label of entity Q23.
      entity_label = items[1]
      # Converting into integers like above

      parent_ids = items[3]
      # NONE if no parent_id
      if parent_ids != "NONE":
        parent_ids = parent_ids.split()
        #  parent_ids = [int(parent_id[1:]) for parent_id in parent_ids]
        parent_ids = tuple(parent_ids)
        entity_id2parents[entity_id] = parent_ids

      # hard code here: filtering
      if not (entity_label.find("Wikimedia project page") >= 0 or
              entity_label.find("metaclass") >= 0 or
              entity_label.find("Wikimedia list article") >= 0 or
              entity_label.find("Wikimedia disambiguation page") >= 0):
        if wiki_title != "NONE":
          entity_id2label[entity_id] = wiki_title
        else:
          entity_id2label[entity_id] = entity_label

      wiki_title2id[wiki_title.lower().strip()] = entity_id
  return wiki_title2id, entity_id2label, entity_id2parents


def collect_qids(path_dir, entities_tsv, confidence_thre=0.0):
  qids_required = {}

  for file_dir, _, filenames in os.walk(path_dir):
      for filename in filenames:
        file_path = os.path.join(file_dir, filename)
        if not file_path.endswith("json"):
          continue
        with open(file_path, 'r') as myfile:
          data = json.loads(myfile.read())
          sent_count = 0
          linked_ent_count = 0
          for item in data:
            ents = item['ents']
            sent_count += 1
            for ent in ents:
              qid = ent[0]
              confidence = float(ent[3])
              if confidence < confidence_thre:
                continue
              qids_required[qid] = 1

  # Obtain parent node of current qids.
  ret = load_wikidata(entities_tsv)
  #  ret = load_wikidata("/Users/yukun/workspace/kg-bert/entities.slimed.tsv")
  _, entity_id2label, entity_id2parents = ret

  qid2descrip = {}
  for qid, _ in qids_required.items():
    if qid not in entity_id2parents:
      continue
    parent_qids = entity_id2parents[qid]
    for parent_qid in parent_qids:
      if parent_qid not in entity_id2label:
        continue
      descrip = entity_id2label[parent_qid]
      qid2descrip[parent_qid] = descrip

  print(f"#used qid: {len(qid2descrip.keys())}")
  return qid2descrip

class Feature(object):
  def __init__(self, input_ids, input_mask, segments_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segments_ids
    self.label_id = label_id
    

def prepare_desrip_ebm(qid2descrip, args):
  # Batching description
  tokenizer = BertTokenizer.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)
  idx = 0
  features = []
  for qid, descrip in qid2descrip.items():
    tokens = tokenizer.tokenize_no_ent(descrip)
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > args.max_seq_length - 2:
      tokens = tokens[:(args.max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    max_seq_length = args.max_seq_length
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    if idx < 5:
      print(f"tokens: {tokens}")
      print(f"input_ids: {input_ids}")
      print(f"segment_ids: {segment_ids}")
      print(f"input_mask: {input_mask}")
      idx += 1
    f = Feature(input_ids, input_mask, segments_ids, qid):
    features.append(f)

  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  ## Required parameters
  parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--ernie_model", default=None, type=str, required=True,
                      help="Ernie pre-trained model")
  parser.add_argument("--entities_tsv", default=None, type=str, required=True,
                      help="entties files where descriptions are stored.")
  ## Other parameters
  parser.add_argument("--max_seq_length",
                      default=10,
                      type=int,
                      help="The maximum total input sequence length after WordPiece tokenization. \n"
                           "Sequences longer than this will be truncated, and sequences shorter \n"
                           "than this will be padded.")
  parser.add_argument("--do_lower_case",
                      default=False,
                      action='store_true',
                      help="Set this flag if you are using an uncased model.")
  parser.add_argument('--threshold', type=float, default=.0)
  
  args = parser.parse_args()

  qid2descrip = collect_qids(args.data_dir, args.entities_tsv, args.threshold)
  prepare_desrip_ebm(qid2descrip, args)
