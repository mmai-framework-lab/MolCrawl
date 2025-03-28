import logging

from rdkit import Chem, RDLogger
from rdchiral.chiral import copy_chirality
from rdkit.Chem.AllChem import AssignStereochemistry

from transformers import AutoTokenizer
from core.base import TrainableTokenizer

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def canonicalize(smiles, isomeric=False, canonical=True, kekulize=False):
    # When canonicalizing a SMILES string, we typically want to
    # run Chem.RemoveHs(mol), but this will try to kekulize the mol
    # which is not required for canonical SMILES.  Instead, we make a
    # copy of the mol retaining only the information we desire (not explicit Hs)
    # Then, we sanitize the mol without kekulization.  copy_atom and copy_edit_mol
    # Are used to create this clean copy of the mol.
    def copy_atom(atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        if atom.GetIsAromatic() and atom.GetNoImplicit():
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            # elif atom.GetSymbol() == 'N':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
            #    new_atom.SetNumExplicitHs(-atom.GetImplicitValence())
            # elif atom.GetSymbol() == 'S':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
        return new_atom

    def copy_edit_mol(mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(""))
        for atom in mol.GetAtoms():
            new_atom = copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
            new_bond = new_mol.GetBondBetweenAtoms(a1, a2)
            new_bond.SetBondDir(bond.GetBondDir())
            new_bond.SetStereo(bond.GetStereo())
        for new_atom in new_mol.GetAtoms():
            atom = mol.GetAtomWithIdx(new_atom.GetIdx())
            copy_chirality(atom, new_atom)
        return new_mol

    smiles = smiles.replace(" ", "")
    tmp = Chem.MolFromSmiles(smiles, sanitize=False)
    tmp.UpdatePropertyCache()
    new_mol = copy_edit_mol(tmp)
    # Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    if not kekulize:
        Chem.SanitizeMol(
            new_mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_PROPERTIES
            | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
            catchErrors=True,
        )
    else:
        Chem.SanitizeMol(
            new_mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE
            | Chem.SanitizeFlags.SANITIZE_PROPERTIES
            | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
            catchErrors=True,
        )

    AssignStereochemistry(new_mol, cleanIt=False, force=True, flagPossibleStereoCenters=True)

    new_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=isomeric, canonical=canonical)
    return new_smiles


def canonicalize_molecule_smiles(
    smiles,
    return_none_for_error=True,
    skip_mol=False,
    sort_things=True,
    isomeric=True,
    kekulization=True,
    allow_empty_part=False,
):
    things = smiles.split(".")
    if skip_mol:
        new_things = things
    else:
        new_things = []
        for thing in things:
            try:
                if thing == "" and not allow_empty_part:
                    raise ValueError("SMILES contains empty part.")

                mol = Chem.MolFromSmiles(thing)
                if mol is None:
                    return None
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                thing_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=False, isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=False, isomericSmiles=isomeric)
                assert thing_smiles is not None
                can_in = thing_smiles
                can_out = canonicalize(thing_smiles, isomeric=isomeric)
                assert can_out is not None, can_in
                thing_smiles = can_out
                if kekulization:
                    thing_smiles = keku_mid = Chem.MolFromSmiles(thing_smiles)
                    assert keku_mid is not None, "Before can: %s\nAfter can: %s" % (can_in, can_out)
                    thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=True, isomericSmiles=isomeric)
            except KeyboardInterrupt:
                raise
            except Exception:
                if return_none_for_error:
                    return None
                else:
                    raise
            new_things.append(thing_smiles)
    if sort_things:
        new_things = sorted(new_things)
    new_things = ".".join(new_things)
    return new_things


def generate_chat(input_text, output_text=None, prefix_chat=None):
    chat = [
        {"role": "user", "content": input_text},
    ]
    if output_text is not None:
        chat.append({"role": "assistant", "content": output_text})
    if prefix_chat is not None:
        chat = prefix_chat + chat
    return chat


def get_chat_content(conversation, tokenize=False):
    if tokenize:
        raise NotImplementedError
    available_roles = ("user", "assistant")
    content = ""
    for idx, item in enumerate(conversation):
        role = item["role"]
        assert role in available_roles, role
        if idx % 2 == 0:
            assert role == "user"
            content += "<s>"
            item_content = "[INST] %s [/INST]" % item["content"]
            content += item_content
        else:
            assert role == "assistant"
            item_content = " %s</s>" % item["content"]
            content += item_content
    return content


class GeneralPrompter(object):

    def __init__(self, apply_chat_template_func, response_split="[/INST]"):
        self.apply_chat_template_func = apply_chat_template_func
        self.response_split = response_split

    def generate_prompt(self, chat, tokenize=False, *args, **kargs) -> str:
        res = self.apply_chat_template_func(chat, tokenize=tokenize, *args, **kargs)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.response_split)[-1].strip()


class MoleculeNatLangTokenizer(TrainableTokenizer):

    def __init__(
        self,
    ):
        TrainableTokenizer.__init__(self)
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.sep_token = "<unk>"
        self.tokenizer.cls_token = "<unk>"
        self.tokenizer.mask_token = "<unk>"

        self.prompter = GeneralPrompter(get_chat_content)

    def __getattr__(self, name):
        """
        If an attribute is not found in this class, try to get it from self.tokenizer.
        """
        if hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _tokenize(self, text: str, add_eos_token: bool = True):
        # there's probably a way to do this with the tokenizer settings
        result = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        if result["input_ids"][-1] != self.tokenizer.eos_token_id and add_eos_token:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def tokenize_dict(self, text: str, canonicalize_smiles: bool = True, max_input_tokens: bool = None):
        tokenized_output = self.tokenizer(
            text["output"],
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]
        tokenized_output.append(self.tokenizer.eos_token_id)

        sample = self.tokenize_text(text["input"], canonicalize_smiles=canonicalize_smiles, max_input_tokens=max_input_tokens)

        sample["output_ids"] = tokenized_output

        return sample

    def tokenize_text(self, text, canonicalize_smiles: bool = True, max_input_tokens: bool = None):
        if canonicalize_smiles:
            real_text = self.canonicalize_smiles_in_text(text)
        else:
            real_text = text

        sample = {"input_text": text}
        chat = generate_chat(real_text, output_text=None)
        full_prompt = self.prompter.generate_prompt(chat)
        sample["real_input_text"] = full_prompt
        tokenized_full_prompt = self._tokenize(full_prompt, add_eos_token=False)
        sample.update(tokenized_full_prompt)
        if max_input_tokens is not None and len(tokenized_full_prompt["input_ids"]) > max_input_tokens:
            sample["input_too_long"] = True

        return sample

    @staticmethod
    def canonicalize_smiles_in_text(
        text, tags=("<SMILES>", "</SMILES>"), keep_text_unchanged_if_no_tags=True, keep_text_unchanged_if_error=False
    ):
        try:
            left_tag, right_tag = tags
            assert left_tag is not None
            assert right_tag is not None

            left_tag_pos = text.find(left_tag)
            right_tag_pos = None
            if left_tag_pos == -1:
                assert right_tag not in text, 'The input text "%s" only contains the right tag "%s" but no left tag"%s"' % (
                    text,
                    right_tag,
                    left_tag,
                )
                return text
            else:
                right_tag_pos = text.find(right_tag)
                assert right_tag_pos is not None, 'The input text "%s" only contains the left tag "%s" but no right tag"%s"' % (
                    text,
                    left_tag,
                    right_tag,
                )
        except AssertionError:
            if keep_text_unchanged_if_no_tags:
                return text
            raise

        smiles = text[left_tag_pos + len(left_tag) : right_tag_pos].strip()
        try:
            smiles = canonicalize_molecule_smiles(smiles, return_none_for_error=False)
        except KeyboardInterrupt:
            raise
        except Exception:
            if keep_text_unchanged_if_error:
                return text

        if smiles is None:
            return ""

        new_text = (
            text[:left_tag_pos]
            + ("" if (left_tag_pos == 0 or text[left_tag_pos - 1] == " ") else " ")
            + left_tag
            + " "
            + smiles
            + " "
            + right_tag
            + " "
            + text[right_tag_pos + len(right_tag) :].lstrip()
        )
        return new_text

    def train(self):
        pass
