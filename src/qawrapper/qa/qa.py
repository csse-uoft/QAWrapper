import torch
if torch.backends.mps.is_available():
    device_str = 'mps'
    mps_device = torch.device(device_str)
    x = torch.ones(1, device=mps_device)
    print("MPS found", x)
else:
    device_str = 'cpu'
    print ("MPS device not found.")
print("Device found: ", device_str)
from transformers import pipeline, AutoModelForQuestionAnswering
from abc import ABC


class QAModel:
    """
    A class that initiates a transformer pipeline and deepset roBERTa squad2 Q&A model.

    ...

    Attributes
    ----------
    model_name : str
        The name of the QA model
    nlp : transformers.pipeline
        A transformer pipeline from the given model name
    model: transformers.AutoModelForQuestionAnswering
        A transformer qa model with the given model name

    """
    model_name = "deepset/roberta-base-squad2"
    # model_name = "deepset/deberta-v3-large-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device_str)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # DebertaForQuestionAnswering



class QA(ABC):
    """
    An abstract class that defines default use cases for the QA knowledge source.

    ...

    Attributes
    ----------
    correct_entity_list : list
        list of entity names that are correct
    context : str
        text to be the context of QA input
    entity : str
        type of entity that questions should target
    QAs : NoneType, dict
        formatted questions and answers
    qa_info : dict
        top QA results
    display_info : dict
        QA results formatted for user display

    Methods
    ----------
    set_questions()
        Set questions on QAs from the question template
    set_answers()
        Run the model and add the outputted information to QAs
    rank_answers()
        Sort QAs by ranking, where the default is by score
    top_answers()
        Add the top ranked answer for each entity to qa_info
    run_qa()
        Do all from setting the question to selecting the top answers
    get_display_info()
        Format top answers to be displayed to users
    fetch_info()
        Fetch all original records of this Q&A

    """

    AGG_PARTIAL = {}  # overridden by qa0 and qa1
    ENTITIES = []   # overridden by qa0 and qa1
    def __init__(self, context, out_entity):
        """Instantiate only if the entity is valid. Raise exception otherwise.

        :param context: text to be the context of QA input
        """
        if out_entity not in self.ENTITIES:
            raise Exception(f"Invalid entity type: {out_entity}")
        self.context = context
        self.QAs = None
        self.qa_info = {}
        self.display_info = {}
        self.out_entity = out_entity

    def set_questions(self):
        """
        Set questions on QAs from the question template
        """
        raise NotImplementedError

    def set_answers(self, ner):
        """
        Run the model and add the outputted information to QAs
        """
        raise NotImplementedError

    def get_agg_score(self, out_entity):
        raise NotImplementedError


    def rank_answers(self):
        """
        Sort QAs by ranking, where the default is by score
        """
        for out_entity in self.QAs:
            self.QAs[out_entity] = sorted(self.QAs[out_entity], key=lambda d: d['score'])

    def top_answers(self):
        """Add the top ranked answer for each out_entity to qa_info

        :return: top answers
        """
        self.rank_answers()
        self.qa_info = {out_entity: self.QAs[out_entity][0] for out_entity in self.QAs}
        return self.qa_info

    def run_qa(self, ner=False):
        """Do all from setting the question to selecting the top answers

        :return: top answers
        """
        self.set_questions()
        self.set_answers(ner)
        self.qa_info = self.top_answers()
        return self.qa_info

    def get_display_info(self):
        """Format top answers to be displayed to users

        :return: formatted QA outputs to be displayed to users
        """
        display_info = {}
        for out_entity in self.qa_info:
            display_info[out_entity] = self.qa_info[out_entity]["answer"]
        self.display_info = display_info
        return display_info

    def fetch_info(self):
        """Fetch all original records of this Q&A

        :return: dictionary of the context and QAs
        """
        return {"context": self.context, "QAs": self.QAs}
