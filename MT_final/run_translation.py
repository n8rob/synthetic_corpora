import torch
import argparse
from trans import *

parser = argparse.ArgumentParser()
parser.add_argument("--mode",
                    type=str,
                    default='play',
                    help="One of the following: {'play', 'document', 'humeval'}"
                    )
parser.add_argument("--src-file",
                    type=str,
                    #required=True,
                    help="file with source sentences to translate"
                    )
parser.add_argument("--model",
                    default='en2ht_transln4.pt',
                    help="path to trained translator"
                    )
parser.add_argument("--vocab",
                    default="en2ht_transln4_VOCAB.pkl",
                    help="path to vocabulary files for trained translator"
                    )
parser.add_argument("--out-file",
                    default="translated_doc.txt",
                    help="name of translation output file"
                    )
parser.add_argument("--test-file",
                    default="haitian_test",
                    help="name of file for test sentences in source language"
                    )
parser.add_argument("--limit",
                    type=int,
                    default=10000,
                    help="how many sentences to translate"
                    )
parser.add_argument("--backwards",
                    type=int,
                    default=0,
                    help="read lines from the back?"
                    )
parser.add_argument("--gpu-num",
                    type=int,
                    default=0,
                    help="which GPU to use"
                    )

args = parser.parse_args()

args.backwards = bool(args.backwards)

print(args)

GPU_NUM = args.gpu_num

def play(model_path=args.model, vocab_pkl=args.vocab):
    # Retrieve entities we need
    model = torch.load(model_path, map_location=torch.device('cpu')).to(torch.device('cuda:{}'.format(GPU_NUM)))
    with open(vocab_pkl,'rb') as f:
        vocab = pkl.load(f)

    while True:
        src_text = input("Enter a sentence in source language / Tape yon fraz an lang sous la\n(Type 'stop' to stop):\t")

        if src_text.lower() == 'stop':
            break

        src = torch.Tensor([0] + [vocab["SRC.vocab.stoi"][word] for word in src_text.split()] + [0]).unsqueeze(0).long().to(torch.device("cuda:{}".format(args.gpu_num)))
        src_mask = (src != vocab["SRC.vocab.stoi"]["<blank>"]).unsqueeze(-2).to(torch.device("cuda:{}".format(args.gpu_num)))
        out = greedy_decode(model, src, src_mask,
                        max_len=60, start_symbol=vocab["TGT.vocab.stoi"]["<s>"])
        # print("haitian:", end="\t")
        # for i in range(0, src.size(1)):
        #     sym = vocab["SRC.vocab.itos"][src[0, i]]
        #     if sym == "</s>": break
        #     print(sym, end =" ")
        # print()
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = vocab["TGT.vocab.itos"][out[0, i]]
            if sym == "</s>": break
            print(sym, end =" ")
        print()
        print()

def translate_sentence(src_sent, model, vocab):
    """
    Translate single sentence and return translation
    """
    src = torch.Tensor([0] + [vocab["SRC.vocab.stoi"][word] for word in src_sent.split()] + [0]).unsqueeze(0).long().to(torch.device("cuda:{}".format(args.gpu_num)))
    src_mask = (src != vocab["SRC.vocab.stoi"]["<blank>"]).unsqueeze(-2).to(torch.device("cuda:{}".format(args.gpu_num)))
    out = greedy_decode(model, src, src_mask,
                        max_len=60, start_symbol=vocab["TGT.vocab.stoi"]["<s>"])
    out_sent = ""
    for i in range(1, out.size(1)):
        sym = vocab["TGT.vocab.itos"][out[0, i]]
        if sym == "</s>": break
        out_sent = out_sent + sym + " "

    return out_sent.strip()

def translate_doc(model_path=args.model, src_file=args.src_file, vocab_pkl=args.vocab, out_file=args.out_file, limit=args.limit, backwards=args.backwards):
    """
    Translate full document and save to new file
    """
    # Retrieve entities we need
    model = torch.load(model_path, map_location=torch.device('cpu')).to(torch.device('cuda:{}'.format(GPU_NUM)))
    with open(vocab_pkl,'rb') as f:
        vocab = pkl.load(f)
    if type(src_file) is str:
        with open(src_file,'r') as f:
            src_sents = f.readlines()
    elif type(src_file) is list:
        src_sents = src_file
    else:
        raise TypeError("Variable src_file must be str or list")

    L = len(src_sents)
    if limit:
        if backwards:
            src_sents = src_sents[-limit:]
        else:
            src_sents = src_sents[:limit]

    L = len(src_sents)
    num_times_to_print = 20
    print_every = L // 20

    print("Translating {}".format(src_file), flush=True)
    out_sents = []
    for i, src_sent in enumerate(src_sents):
        try:
            out_sents.append(translate_sentence(src_sent, model, vocab) + '\n')
        except:
            print("err")
        
        print(i, flush=True)

        #if i % print_every == 0:
        #    percent_to_display = round((i+1)/L)
        #    print("{}%".format(percent_to_display), flush=True, end=" ")

    print()
    print("Translation lengths", len(src_sents), len(out_sents), flush=True)

    if out_file:
        with open(out_file, 'w') as f:
            f.writelines(out_sents)

    return out_sents

def human_eval(test_file=args.test_file, baseline_model="ht2en_transln10", challenger_names=[], sents_per_mod=50, out_file="hum_evals.txt", key_file="hum_key.txt"):
    """
    Read test file, use models, to translate, and print out translations for human eval to test file
    """
    # write to output files
    with open(out_file, 'w') as f:
        f.write("\n")
    with open(key_file, 'w') as f:
        f.write("\n")

    # read in source lines
    with open(test_file, 'r') as f:
        all_test_lines = f.readlines()

    # One of the models is the baseline model
    bl_mod_file = baseline_model + ".pt"
    bl_vocab_file = baseline_model + "_VOCAB.pkl"

    for i, model_name in enumerate(challenger_names):
        # Get arguments necessary to call translate_doc
        mod_file = model_name + ".pt"
        vocab_file = model_name + "_VOCAB.pkl"
        test_lines = all_test_lines[i*sents_per_mod:(i+1)*sents_per_mod]
        # translate
        baseline_trans = translate_doc(model_path=bl_mod_file, src_file=test_lines, vocab_pkl=bl_vocab_file, out_file=False, limit=False, backwards=False)
        challenger_trans = translate_doc(model_path=mod_file, src_file=test_lines, vocab_pkl=vocab_file, out_file=False, limit=False, backwards=False)
        assert len(baseline_trans) == len(challenger_trans)
        # figure out which order to print them in (randomly)
        random_bit = np.random.randint(2) 
        if random_bit:
            first_set, second_set = baseline_trans, challenger_trans
            with open(key_file, 'a') as f:
                f.write("The first one is the baseline; the second is the challenger.\n")
        else:
            first_set, second_set = challenger_trans, baseline_trans
            with open(key_file, 'a') as f:
                f.write("The first one is the challenger; the second is the baseline.\n")
        with open(key_file, 'a') as f:
            f.write("\n##########################################\n\n")
        # Then write lines to actual out_file
        with open(out_file, 'a') as f:
            for j, (first, second) in enumerate(zip(first_set, second_set)):
                f.write('(' + str(j+1) + ')\t' + test_lines[j].strip() + '\n(A)\t' + first.strip() + "\n(B)\t" + second + '\n\n~~~\n\n')
            f.write("\n\n###################################################################################\n\n")

    print("done")
    return

def create_human_eval():
    challenger_models = ["ht2en_biling_transln9",
                         "ht2en_augm_transln6",
                         "ht2en_augm_transln8",
                         "ht2en_augm_transln11"
                         ]
    human_eval(challenger_names=challenger_models)
    return
    

if __name__ == "__main__":

    if args.mode == 'play':
        play()
    elif args.mode == 'document':
        translate_doc()
    elif args.mode == 'humeval':
        create_human_eval()
    pass
