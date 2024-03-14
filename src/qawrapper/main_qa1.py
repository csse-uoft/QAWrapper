from .qa1 import QA1
if __name__ == '__main__':
    description = "St.Mary's Church is focused on raising well rounded and spiritually sound youths and teenagers, equipping them with what it takes to overcome the daily pressures of the society. They have separate meeting during our services."
    entity = "client"
    qa1 = QA1(description, entity)
    qa1.update_given_ner(ent_type="outcome", givens={"well rounded and spiritually sound youths and teenagers": 0.5})
    qa1.update_given_ner(ent_type="program_name", givens={"St.Mary's Church": 0.5})
    print(qa1.given)
    output_dic = qa1.run_qa(ner=True)
    print(output_dic)