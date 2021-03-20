import json
import copy
import random

questions_dict = {
    "What was Jill's mother's face burned by?": "What was Jill's mother's face burned with?",
    "What year did a lynch mod muder someone who they believed to be a warlock?": "In what year did a lynch change someone they believed to be a warlock?",
    "Whose corpse is in the morgue?": "What corpse is in the morgue?",
    "Whose corpse does Liza see?": "What corpse does Liza see?",
    "What room number is investigated?": "What room number is under investigation?",
    "Who transports himself to Bermuda?": "Who ships to Bermuda?",
    "Where did the sword appear to be": "Where did the sword appear?",
    "Who uses scientific skill over trickery?": "Who uses scientific competence rather than deception?",
    "Who is Merlin's pet owl?": "Who is Merlin's owl?",
    "Who does Merlin appoint as Arthur's teacher?": "Who does Merlin appoint as Arthur's teacher?",
    "Who is elated to find that Arthur is king?": "Who is thrilled to find out that Arthur is king?",
    "Who does Gramp's try to say goodbye to?": "Who is Gramp trying to say goodbye to?",
    "Who calls Gramps and Pud from the other side?": "Who's calling Gramps and Pud on the other side?",
    "Who does Gramps shoot in order to prove his power?": "Who is Gramps shooting at to prove his power?",
    "Brink kills Granny Nellie after she finishes doing what?": "Brink kills Granny Nellie after he's done doing what?",
    "Who doees Mr Brink dare to climb the tree?": "Who does Mr. Brink dare to climb the tree?",
    "Who calls them from the brilliant light?": "Who calls them bright light?",
    "What was  blocking Pud's way?": "What was blocking Pud's path?",
    "What does Gramps trick Mr. Brink into doing?": "What does Gramps make Mr. Brink do?",
    "Who does Gramps claim is trapped in his apple tree?": "Who does Gramps think is trapped in his apple tree?",
    "Who orders Mr. Brink off the property?": "Who orders Mr. Brink to leave the property?",
    "What is Pud's Aunt's name?": "What is the name of Pud's aunt?",
    "Who arranges for the local sheriff to commit Gramps?": "Who makes the local sheriff commit Gramps?",
    "What is a part of life?": "What is a part of life?",
    "What prototype is under heavy guard?": "Which prototype is under close surveillance?",
    "Who has got the wind of the operation and is already hot on Gant's tail?": "Who is in charge of the operation and who has already warmed Gant's tail?",
    "Who engages Gant in a dogfight?": "Who engages Gant in aerial combat?",
    "What is the name of the man Kitty has an affair with?": "What is the name of the man Kitty is having an affair with?",
    "What room does Kitty volunteer in when they arrive in China?": "What room does Kitty volunteer in when she arrives in China?",
    "What disease is Walter there to treat?": "What disease does Walter have to treat?",
    "How long after Walter dies does Kitty run into Townsend?": "How long after Walter Kitty's death does she run into Townsend?",
    "where the doctor is stationed in a government lab studying infectious diseases?": "where the doctor is stationed in a government laboratory studying infectious diseases?",
    "What happens that causes Kitty & Walter to reignite the love they had for each other?": "What makes Kitty and Walter rekindle the love they had for each other?",
    "who does kitty meet?": "who meets kitty?",
    "Who assigns the police to investigate the robberies?": "Who assigns the police to investigate the thefts?",
    "Who is revealed to be the \"Mastermind\"?": "Who turns out to be the \"Mastermind\"?",
    "who is guiding the group of three dimwitted criminals ?": "Who is leading the group of three stupid criminals?",
    "Who does the police academy do battle with ?": "Who does the police academy fight with?",
    "Did Nick get anywhere by distributing the flyers?": "Did Nick get somewhere giving out the flyers?",
    "Who has been unwittingly leaking information during his daily meetings with the mayor?": "Who unintentionally disclosed information during their daily meetings with the mayor?",
    "What gang does the police academy find and do battle with?": "What gang does the police academy find and fight with?",
    "Who chases the leader?": "Who is suing the leader?",
    "What does Harris shout as he floats up in the air?": "What is Harris screaming as he floats in the air?",
    "What is the name of the gang the police are looking for?": "What is the name of the gang wanted by the police?",
    "Who nabbed the diamond?": "Who caught the diamond?",
    "Who is the mastermind ?": "Who is the brain?",
    "What is given to honor the officers' bravery?": "What is given to honor the bravery of the officers?",
    "Where does Nick deduce the robberies are occurring?": "Where does Nick deduce the thefts are happening?",
    "What is the name of the gang Nick is trying to take down?": "What is the name of the gang Nick is trying to destroy?",
    "Which  technology will eventually grant Castle unlimited power over the populace?": "What technology will ultimately grant Castle unlimited power over the people?",
    "What is the name of the activist organization in the movie?": "What is the name of the militant organization in the film?",
    "In the movie, Tillman convinces Castle's men to deactivate what?": "In the movie, Tillman convinces Castle's men to turn off what?",
    "Castle amasses a fortune that surpasses that of?": "Castle is amassing a fortune that surpasses that of?",
    "How many matches does an inmate have to survive to gain freedom?": "How many matches does an inmate have to survive to gain his freedom?",
    "How much does Holly make for each weekly trip to Sing Sing prison?": "How much does Holly earn for each weekly trip to Sing Sing Prison?",
    "Who realizes that he is in love with Holly and proposes to her?": "Who realizes he's in love with Holly and proposes to her?",
    "What city does Holly live in?": "What town does Holly live in?",
    "With what does Vasquez threaten the other characters with?": "What does Vasquez threaten the other characters with?",
    "What country was The Silver Queens destination?": "What country was The Silver Queens in?",
    "Who patches the airliner's oil leak?": "Who fixes the airliner's oil leak?",
    "Who eventually discovers Crimp's dead body?": "Who ultimately finds Crimp's corpse?",
    "Who does Pink take back to his hotel room?": "Who is Pink bringing back to her hotel room?",
    "What does Pink lose her mind to ?": "What is Pink losing his mind?",
    "Where did Pink's father die in combat?": "Where did Pink's father die in battle?",
    "What kind of doll is Pink depicted as?": "What kind of doll does Pink represent?",
    "what is the name of the captive girl?": "What's the name of the captive girl?",
    "Who agrees to meet with Dane?": "Who agrees to meet Dane?",
    "where are the vampires sailing to ?": "Where are the vampires sailing?",
    "what does stella try and convince others exists?": "What is Stella trying to convince that the others exist?",
    "what do the hunters enter?": "What are the hunters getting in?",
    "what does agent norris do to the captive girl?": "what is agent norris doing to the captive?",
    "what are the hunters hunting?": "What do hunters hunt?",
    "What incinerates several vampires?": "What cremates several vampires?",
    "Who does Mona sell Thumbelina to?": "Who is Mona selling Thumbelina to?",
    "Who is Thumbelina on a quest to find?": "Who is Thumbelina looking to find?",
    "Who thaws out the prince?": "Who thaws the prince?",
    "Who becomes the princess of the land to the cheers of the little people?": "Who becomes the princess of the country to the cheers of the little people?",
    "Who duels with the toad?": "Who's fighting with the toad?",
    "Where does the bird take Thumbelina?": "Where is the bird taking Thumbelina?",
    "Who discovers Mona's wicked plan?": "Who finds out about Mona's wicked plan?",
    "Who does Thumbelina marry?": "Who is Thumbelina marrying?",
    "Who brings the swallow a blanket?": "Who brings a blanket to the swallow?"
}

def get_questions():
    # from google.cloud import translate_v2 as translate

    # translate_client = translate.Client()

    with open("datasets/oodomain_train/relation_extraction_orig") as f:
        for line in f:
            line_str = line.strip()
            break
    '''
    json_file = json.loads(line_str)
    questions = {}
    for elem in json_file["data"]:
        for p in elem["paragraphs"]:
            for qas in p["qas"]:
                pass
                # fr_result = translate_client.translate(qas["question"], source_language="en", target_language="fr")
                # result = translate_client.translate(fr_result, source_language="fr", target_language="en")
                # questions[qas["question"]] = result

    with open("relex_questions.tsv", "w+") as f:
        f.write(json.dumps(questions))
    '''

    json_file = json.loads(line_str)
    questions = []
    for elem in json_file["data"]:
        for p in elem["paragraphs"]:
            for qas in p["qas"]:
                questions.append(qas["question"])

    with open("relex_questions.tsv", "w+") as f:
        f.write("\n".join(questions))


def main():
    with open("datasets/oodomain_train/relation_extraction_orig") as f:
        for line in f:
            line_str = line.strip()
            break

    questions_dict = {}
    with open("relex_questions.tsv") as f:
        idx = 1
        for line in f:
            tokens = line.strip().split("\t")
            print(idx, tokens)
            assert(len(tokens) == 2)
            questions_dict[tokens[0]] = tokens[1]
            idx += 1


    json_file = json.loads(line_str)
    print("   "+ str(len(json_file["data"])))
    json_add = []
    for elem in json_file["data"]:
        elem_copy = copy.deepcopy(elem)
        for p in elem_copy["paragraphs"]:
            for qas in p["qas"]:
                if qas["question"] in questions_dict:
                    qas["question"] = questions_dict[qas["question"]]
        json_add.append(elem_copy)
    json_file["data"] += json_add
    print("   "+ str(len(json_file["data"])))

    json_add_2 = []
    for elem in json_file["data"]:
        for _ in range(20):
            elem_copy = copy.deepcopy(elem)
            for p in elem_copy["paragraphs"]:
                for qas in p["qas"]:
                    min_ans_start = float("inf")
                    for a in qas["answers"]:
                        min_ans_start = min(min_ans_start, a["answer_start"])
                shift = random.randint(int(0.2*min_ans_start), int(0.8*min_ans_start))
                print(shift)
                tokens = p["context"].split()
                tokens = tokens[shift:]
                p["context"] = " ".join(tokens)
                for qas in p["qas"]:
                    for a in qas["answers"]:
                        a["answer_start"] -= shift
            json_add_2.append(elem_copy)
    json_file["data"] += json_add_2
    print("   "+ str(len(json_file["data"])))

    with open("datasets/oodomain_train/relation_extraction", "w+") as f:
        f.write(json.dumps(json_file))



if __name__ == "__main__":
    main()
    # get_questions()