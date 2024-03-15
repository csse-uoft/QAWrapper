from .qa.qa import QA, QAModel
from .qa import qa_generate_layer1
import pandas as pd
import os, sys
from collections import defaultdict

class QA1(QA):
    """
    A class inherited from QA that deals with the layer 1 qa knowledge source use cases.

    ...

    Attributes
    ----------
    default_questions : dict
        QA1 default questions
    given : dict
        Given keywords for each entity

    Methods
    -------
    update_given_ner(ent_type, givens)
        Update the given keywords for an entity type in the instance variable given
    make_questions()
        Make questions for each target entity using givens
    set_questions(ner)
        prepare the results formatting and set questions
    set_answers(entity)
        run the qa model to get answers
    get_agg_scores()
        get aggregate scores for a particular entity

    """
    AGG_PARTIAL = pd.read_csv(os.path.dirname(os.path.abspath(sys.modules[QA.__module__].__file__))+"/aggregate_scores/layer1_partial2_aggregate_comb.csv", index_col=0).to_dict('index')
    ENTITIES = [] #will be overwritten

    def __init__(self, context, out_entity):
        super().__init__(context, out_entity)
        self.given = defaultdict(dict)

    # def update_given_ner(self, ent_type: str, givens: dict[object: float]):
    def update_given_ner(self, in_entity, givens):
        """Update the given keywords for an entity type in the instance variable given

        :param ent_type: entity type, abbreviated by two characters, of the givens
        :param givens: given keywords to be updated
        """
        if in_entity in self.ENTITIES:
            for g in givens:
                if (g not in self.given[in_entity]) or (self.given[in_entity][g] < givens[g]):
                    self.given[in_entity][g] = givens[g]
        else:
            print("given type is invalid.", in_entity, self.given)

    def make_questions(self):
        """
        Make questions for each target entity using givens

        :return: all questions
        """
        all_Qs = {}
        for in_entity in self.given:
            all_Qs[in_entity] = {}
            for keyword in self.given[in_entity]:
                questions = qa_generate_layer1.get_q_by_entity(in_entity, str(keyword))
                all_Qs[in_entity][str(keyword)] = questions
        return all_Qs

    def set_questions(self):
        """
        prepare the results formatting and set questions
        """
        new_QAs = {}
        all_Qs = self.make_questions()
        for in_entity in all_Qs.keys():
            for keyword in all_Qs[in_entity]:
                if not self.out_entity in all_Qs[in_entity][keyword].keys():
                    continue
                try:
                    new_QAs[self.out_entity] = []
                    for question in all_Qs[in_entity][keyword][self.out_entity]:
                        new_qa = {"keyword": str(keyword), "giv_ent": in_entity,
                                  "giv_score": self.given[in_entity][keyword],
                                  "question": question, "answer": None, "start": None, "end": None, "score": None}
                        new_QAs[self.out_entity].append(new_qa)
                except KeyError as e:
                    print("Error....")
                    print(type(e), e)
                    return {}
        self.QAs = new_QAs

    def set_answers(self, ner):
        """
        Run the model and add the outputted information to QAs
        :param ner: boolean that tells whether this result will be ner or phrase data level hypothesis
        """
        for in_entity in self.QAs:
            for qa in self.QAs[in_entity]:
                QA_input = {
                    'question': qa["question"],
                    'context': self.context
                }
                res = QAModel.nlp(QA_input)
                qa["answer"] = res["answer"]
                qa["start"] = res["start"]
                qa["end"] = res["end"]
                if ner:
                    out_entity = qa["giv_ent"] + "_" + in_entity
                    qa["score"] = res["score"] * 0.3 + self.get_agg_score(out_entity) * 0.5 + qa["giv_score"] * 0.2
                else:
                    qa["score"] = res["score"]

    def get_agg_score(self, out_entity):
        """
        get aggregate scores for a particular entity
        :param entity: entity type that we want to get aggregate score for
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


