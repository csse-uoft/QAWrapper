from src.qa.qa import QA, QAModel
from src.qa import qa_generate_layer1
import pandas as pd
import os
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
    AGG_PARTIAL = pd.read_csv("src/qa/aggregate_scores/layer1_partial2_aggregate_comb.csv", index_col=0).to_dict('index')
    ENTITIES = [
        "program_name",
        "client",
        "need_satisfier",
        "outcome",
        "catchment_area"
    ]

    def __init__(self, context, entity):
        super().__init__(context, entity)
        self.given = defaultdict(dict)

    # def update_given_ner(self, ent_type: str, givens: dict[object: float]):
    def update_given_ner(self, ent_type, givens):
        """Update the given keywords for an entity type in the instance variable given

        :param ent_type: entity type, abbreviated by two characters, of the givens
        :param givens: given keywords to be updated
        """
        if ent_type in self.ENTITIES:
            for g in givens:
                if (g not in self.given[ent_type]) or (self.given[ent_type][g] < givens[g]):
                    self.given[ent_type][g] = givens[g]
        else:
            print("given type is invalid.", ent_type, self.given)

    def make_questions(self):
        """
        Make questions for each target entity using givens

        :return: all questions
        """
        all_Qs = {}
        for giv_ent_type in self.given:
            all_Qs[giv_ent_type] = {}
            for keyword in self.given[giv_ent_type]:
                questions = qa_generate_layer1.get_q_by_entity(giv_ent_type, str(keyword))
                all_Qs[giv_ent_type][str(keyword)] = questions
        return all_Qs

    def set_questions(self):
        """
        prepare the results formatting and set questions
        """
        new_QAs = {}
        all_Qs = self.make_questions()
        for giv_ent_type in all_Qs.keys():
            for keyword in all_Qs[giv_ent_type]:
                if not self.entity in all_Qs[giv_ent_type][keyword].keys():
                    continue
                try:
                    new_QAs[self.entity] = []
                    for question in all_Qs[giv_ent_type][keyword][self.entity]:
                        new_qa = {"keyword": str(keyword), "giv_ent": giv_ent_type,
                                  "giv_score": self.given[giv_ent_type][keyword],
                                  "question": question, "answer": None, "start": None, "end": None, "score": None}
                        new_QAs[self.entity].append(new_qa)
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
        for entity in self.QAs:
            for qa in self.QAs[entity]:
                QA_input = {
                    'question': qa["question"],
                    'context': self.context
                }
                res = QAModel.nlp(QA_input)
                qa["answer"] = res["answer"]
                qa["start"] = res["start"]
                qa["end"] = res["end"]
                if ner:
                    entity_code = qa["giv_ent"] + "_" + entity
                    qa["score"] = res["score"] * 0.3 + self.get_agg_score(entity_code) * 0.5 + qa["giv_score"] * 0.2
                else:
                    qa["score"] = res["score"]

    def get_agg_score(self, entity):
        """
        get aggregate scores for a particular entity
        :param entity: entity type that we want to get aggregate score for
        :return: precision of the entity type given
        """
        return self.AGG_PARTIAL[entity]['precision']

    # def rank_answers(self, rank=None):
    #     if rank is None:
    #         super().rank_answers()
    #     else:
    #         resorted_QAs = {}
    #         for entity in self.QAs:
    #             resorted_QAs[entity] = [self.QAs[entity][i] for i in rank[entity]]
    #         self.QAs = resorted_QAs


if __name__ == '__main__':
    description = "St.Mary's Church is focused on raising well rounded and spiritually sound youths and teenagers, equipping them with what it takes to overcome the daily pressures of the society. They have separate meeting during our services."
    entity = "client"
    qa1 = QA1(description, entity)
    qa1.update_given_ner(ent_type="outcome", givens={"well rounded and spiritually sound youths and teenagers": 0.5})
    qa1.update_given_ner(ent_type="program_name", givens={"St.Mary's Church": 0.5})
    print(qa1.given)
    output_dic = qa1.run_qa(ner=True)
    print(output_dic)