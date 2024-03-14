from .qa0 import QA0
if __name__ == '__main__':
    description = "We are focused on raising well rounded and spiritually sound youths and teenagers, equipping them with what it takes to overcome the daily pressures of the society. They have separate meeting during our services."
    entity = "client"
    qa0 = QA0(description, entity)
    output_dic = qa0.run_qa(ner=True)
    print(output_dic)

