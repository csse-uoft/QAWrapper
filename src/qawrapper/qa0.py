from .qa.qa import QA, QAModel
import pandas as pd
import os, sys

class QA0(QA):
    """
    A class inherited from QA that deals with the layer 0 qa knowledge source use cases.

    ...

    Attributes
    ----------
    default_questions : dict
        QA0 default questions

    Methods
    -------
    set_questions()
        prepare the results formatting and set questions

    set_answers(adjust_score)
        run the qa model to get answers

    get_agg_score(out_entity)
        get aggregate scores for a particular entity

    """
    default_questions = {"program_name": ["What is the program name?",
                                     "What is this called?",
                                     "Who is the provider?",
                                     "Who provides services?",
                                     "Who offers services?",
                                     "Who offers the program?",
                                     "What is the program?"],
                        "client": ["Who is receiving services?",
                                "Who is receiving the service?",
                                "Who are services offered for?",
                               "Who is this program offered to?",
                               "Who is the beneficiary?",
                               "Who is the program targeting?",
                               "Who is this program for?",
                               "Who is this program for?",
                               "Who is this for?",
                               "Who is eligible for services?",
                               "Who is eligible for the service?"],
                    "need_satisfier": ["What are provided?",
                                       "What services are offered?",
                                       "What programs are provided?",
                                       "List all that is offered by the program."],
                    "outcome": ["What does this aim to do?",
                                     "What does this contribute to?",
                                     "What does this achieve?",
                                     "How does this help them?",
                                     "What is the goal of this program?",
                                     "What does this satisfy?"],
                    "catchment_area": ["Where is this offered?",
                                       "What is the location?",
                                       "Where does this apply to?",
                                       "Where is this happening at?"]}
    ENTITIES = list(default_questions.keys())

    AGG_PARTIAL = pd.read_csv(os.path.dirname(os.path.abspath(sys.modules[QA.__module__].__file__))+"/aggregate_scores/layer0_partial2_aggregate.csv", index_col=0).to_dict('index')

    def __init__(self, context, out_entity=None):
        super().__init__(context, out_entity)

    def set_questions(self):
        """
        prepare the results formatting and set questions
        """
        questions = self.default_questions
        new_QAs = {}
        try:
            new_QAs[self.out_entity] = []
            for question in questions[self.out_entity]:
                new_qa = {"question": question, "answer": None,
                          "start": None, "end": None, "score": None}
                new_QAs[self.out_entity].append(new_qa)
        except KeyError as e:
            return e
        self.QAs = new_QAs

    def set_answers(self, adjust_score):
        """
        Run the model and add the outputted information to QAs
        :param ner: boolean that tells whether this result will be ner or phrase data level hypothesis
        """
        for out_entity in self.QAs:
            for qa in self.QAs[out_entity]:
                QA_input = {
                    'question': qa["question"],
                    'context': self.context
                }

                res = QAModel.nlp(QA_input)
                qa["answer"] = res["answer"]
                qa["start"] = res["start"]
                qa["end"] = res["end"]
                if adjust_score:
                    qa["score"] = res["score"] * 0.3 + self.get_agg_score(out_entity) * 0.7
                else:
                    qa["score"] = res["score"]

    def get_agg_score(self, out_entity):
        """
        get aggregate scores for a particular entity
        :param out_entity: entity type that we want to get aggregate score for
        :return: precision of the entity type given
        """
        return self.AGG_PARTIAL[out_entity]['precision']

    # def rank_answers(self, rank=None):
    #     if rank is None:
    #         super().rank_answers()
    #     else:
    #         resorted_QAs = {}
    #         for out_entity in self.QAs:
    #             resorted_QAs[out_entity] = [self.QAs[out_entity][i] for i in rank[out_entity]]
    #         self.QAs = resorted_QAs

    def top_answers(self):
        self.rank_answers()
        return {key: self.QAs[key][0] for key in self.QAs}

    def fetch_info(self):
        return {"context": self.context, "QAs": self.QAs}


