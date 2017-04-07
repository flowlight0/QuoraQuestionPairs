import sys
import pandas
import re
from csv import QUOTE_ALL
from Levenshtein import distance, matching_blocks, editops

def fix(q):
    return re.findall(r"[\w\d][\w\d'-]*", q.lower(), re.UNICODE)

def process(stopwords, anchors, queue, seen=None):
    if not queue:
        return seen
    if seen is None:
        seen = set()
    q = tuple([w for w in queue.pop(0) if w not in stopwords])
    # q = queue.pop(0)
    while q in seen:
        if not queue:
            return seen
        q = queue.pop(0)
    seen.add(q)

    good_rules = []
    for i, w in enumerate(q):
        if w not in anchors:
            continue
        for rule in anchors[w]:
            apply = True
            for j in xrange(len(rule[0])):
                if i + j >= len(q) or rule[0][j] != q[i + j]:
                    apply = False
                    break
            if apply:
                good_rules.append((i, rule))

    for gr in good_rules:
        q1 = list(q)
        q1[gr[0]:gr[0] + len(gr[1][0])] = list(gr[1][1])
        q1 = tuple(q1)
        if q1 not in seen:
            queue.append(q1)
    return process(stopwords, anchors, queue, seen)

def build_anchors(rules):
    anchors = dict()
    for r in rules:
        if r[0][0] not in anchors:
            anchors[r[0][0]] = list()
        anchors[r[0][0]].append(r)
    return anchors

def main():
    df = pandas.read_csv(sys.argv[1], quoting=QUOTE_ALL)
    df["question2"].fillna("", inplace=True)
    stopwords = set(['a', 'the'])
    # rules = [(['good'], ['great']), (['how', 'can', 'i'], ['how', 'to']), (['what', 'should', 'i', 'do', 'to'], ['how', 'to']), (['how', 'do', 'i'], ['how', 'to']), (['how', 'can', 'you', 'make'], ['what', 'can', 'make']), (['how', 'do', 'we'], ['how', 'to']), (['what', 'are', 'some', 'examples', 'of'], ['what', 'are', 'some', 'of']), (['that', 'can', 'be', 'make'], ['made']), (['can', 'we', 'ever'], ['is', 'it', 'possible', 'to']), (['how', 'to', 'learn', 'to'], ['how', 'to']), (['how', 'i', 'can'], ['how', 'to']), (["what's"], ['what', 'is']), (['high', 'salary', 'income'], ['high', 'income']), (['high', 'paying', 'jobs'], ['high', 'income', 'jobs']), (['what', 'are', 'some', 'of'], ['what', 'are', 'some']), (['what', 'are', 'some'], ['what', 'is']), (['which', 'is'], ['what', 'is']), (['movies'], ['movie']), (['how', 'does'], ['how', 'do']), (['what', 'should', 'i', 'do'], ['what', 'to', 'do']), (['can', 'i'], ['how', 'to']), (['how', 'do', 'you', 'think', 'of'], ['what', 'do', 'you', 'think', 'of']), (['highest'], ['greatest']), (['can', 'hold'], ['holds']), (['cause', 'me', 'to', 'have'], ['cause']), (['can', 'excessive', 'amounts', 'of', 'vitamin'], ['can', 'vitamin']), (['rohingya', 'muslims'], ['rohingya', 'people']), (['oitnb'], ['orange', 'is', 'the', 'new', 'black']), (['season', 'of', 'orange', 'is', 'new', 'black'], ['orange', 'is', 'new', 'black']), (['nobody'], ['no', 'one']), (['earphones'], ['earphone']), (['which', 'are'], ['what', 'is']), (['engineering', 'fields'], ['field', 'of', 'engineering']), (['why', 'does'], ['why', 'do']), (['families'], ['family']), (['how', 'do', 'you'], ['how', 'to']), (['my', 'macbook', 'pro'], ['mac', 'laptop']), (['are', 'there', 'any'], ['what', 'is']), (['become', 'more'], ['become']), (['real', 'estate', 'sector'], ['real', 'estate', 'market']), (['what', 'will', 'be', 'impact', 'of'], ['what', 'are', 'effects', 'of']), (['demonitization', 'of'], ['scrapping', 'of']), (['rupees'], ['rupee']), (['what', 'is', 'easiest', 'way', 'to'], ['how', 'to']), (['what', 'is', 'best', 'way', 'to'], ['how', 'to']), (['why', 'do'], ['why']), (['what', 'are', 'reasons', 'that'], ['why']), (['hate'], ['dislike']), (['sufficient'], ['enough']), (['universities'], ['university']), (['university'], ['school']), (['how', 'to', 'get', 'started', 'to'], ['how', 'to']), (['computer', 'security'], ['information', 'security']), (['earn', 'money'], ['earn']), (['earn', 'on'], ['earn', 'from']), (['what', 'can', 'i', 'do', 'to'], ['how', 'to']), (['how', 'can', 'we'], ['how', 'to']), (['care', 'for'], ['worry', 'about']), (['worried'], ['worry']), (['why', 'are', 'we'], ['why', 'do', 'we']), (['opinions'], ['opinion']), (['two-month-old'], ['months']), (['computer', 'science'], ['cs']), (['online', 'test'], ['test']), (['hiding', 'their', 'intelligence'], ['playing', 'dumb']), (['people'], ['person']), (['is', 'disrupting'], ['will', 'disrupt']), (['how', 'can', 'you'], ['how', 'to']), (['recover', 'your'], ['recover']), (['recover', 'my'], ['recover']), (['my', 'mental', 'illness'], ['mental', 'illness']), (['diagnose', 'and', 'medicate'], ['diagnose']), (['what', 'are'], ['what', 'is']), (['how', 'to', 'choose', 'journal', 'to', 'publish'], ['where', 'do', 'i', 'publish']), (['what', 'is', 'your', 'creative'], ['what', 'is', 'your']), (['resolutions'], ['resolution']), (['how', 'many', 'months'], ['how', 'long']), (['how', 'much', 'time'], ['how', 'long']), (['developing', 'android', 'apps'], ['android', 'app', 'development']), (['gain', 'knowledge', 'in'], ['learn']), (['in', 'clash', 'of', 'clans'], ['clash', 'of', 'clans']), (['traffic', 'on', 'your', 'website'], ['traffic', 'for', 'website']), (['protagonists'], ['characters']), (['films', 'and', 'tv', 'shows'], ['films']), (['how', 'do', 'i', 'to'], ['how', 'to']), (['if', 'there', 'will', 'be', 'war'], ['if', 'war', 'starts']), (['deal', 'with', 'fear'], ['reduce', 'fear']), (['reduce', 'your'], ['reduce']), (['how', 'much', 'funds'], ['how', 'much', 'money']), (['with', 'bachelor', 's', 'degree'], ['for', 'bachelors']), (['job', 'possibilities', 'exist'], ['jobs', 'are', 'available']), (['to', 'self', 'study'], ['to', 'study']), (['to', 'understand'], ['to', 'study']), (['best'], ['great']), (['how', 'and', 'why'], ['why', 'and', 'how']), (['her', 'him'], ['your', 'ex']), (['one', 'or', 'two', 'years'], ['two', 'years']), (['excuse', 'to', 'explain'], ['answer']), (['for', 'job', 'interview'], ['in', 'job', 'interview']), (['what', 'do', 'countries', 'do', 'to'], ['how', 'do', 'countries']), (['how', 'should', 'countries'], ['how', 'do', 'countries']), (['why', 'does'], ['why', 'is']), (['what', 'makes'], ['why', 'is']), (["you've", 'ever'], ['you', 'have']), (['delicious', 'dish'], ['interesting', 'foods']), (['my', 'concentration'], ['concentration']), (['improve'], ['increase']), (['how', 'can', 'one'], ['how', 'to']), (['pros', 'and', 'cons', 'of', 'having'], ['pros', 'and', 'cons', 'of']), (['what', 'would', 'be', 'best', 'way', 'to'], ['how', 'to']), (['how', 'did', 'you'], ['how', 'to']), (['stop', 'smoking'], ['quit', 'smoking']), (['quit', 'quit', 'smoking'], ['quit', 'smoking']), (['homes'], ['buildings']), (['mobile', 'phones'], ['mobile', 'phone']), (['mobile', 'phone'], ['phone']), (['under'], ['below']), (['what', 'year', 'did'], ['when', 'was']), (['books'], ['book']), (['book', 'you', 'have', 'ever', 'read'], ['book', 'ever', 'written']), (['how', 'to', 'become'], ['how', 'to', 'be']), (['write', 'letter', 'to'], ['write', 'to']), (['mr', 'narendra'], ['narendra']), (['what', 'is', 'your', 'theories', 'about'], ['what', 'do', 'you', 'think', 'about']), (['what', 'should', 'someone', 'do', 'to'], ['how', 'to']), (['depression', 'and', 'anxiety'], ['anxiety']), (['how', 'to', 'overcome'], ['how', 'to', 'avoid']), (['how', 'to', 'keep', 'motivated', 'when', 'you'], ['how', 'to', 'keep', 'motivation', 'to']), (['learn', 'new', 'language'], ['learn', 'language']), (['why', 'should', 'one', 'dislike'], ['why', 'people', 'dislike']), (['book', 'i', 'should', 'read'], ['book', 'you', 'have', 'read']), (['self-help'], ['self', 'help']), (['top'], ['best']), (['how', 'should', 'i'], ['how', 'to']), (['how', 'could', 'start'], ['how', 'to', 'start']), (['cricketers'], ['cricket', 'players']), (['on', 'tea', 'break'], ['during', 'tea', 'break']), (['really'], ['actually']), (['have', 'you', 'ever'], ['do', 'you']), (['regretted'], ['regret']), (['what', 'would'], ['what', 'will']), (['what', 'is', 'minimum', 'requirements', 'to'], ['how', 'to']), (['join', 'mit'], ['enter', 'mit']), (['too', 'late', 'for', 'me'], ['too', 'late']), (['how', 'we', 'can'], ['how', 'can', 'we']), (['is', 'it', 'possible', 'to'], ['how', 'can', 'we']), (['being'], ['to', 'be']), (['so', 'you'], ['and']), (['how', 'do', 'i', 'start'], ['how', 'to']), (['selling'], ['sell']), (['place', 'for', 'sex'], ['place', 'to', 'have', 'sex']), (['cultures'], ['culture']), (['major', 'differences'], ['difference']), (['to', 'recommend', 'foreigners', 'to', 'visit'], ['to', 'visit']), (['true'], ['good']), (['great', 'lesson'], ['most', 'important', 'lesson']), (['lesson', 'you', 'have', 'learned', 'from', 'life'], ['lesson', 'in', 'life']), (['favourite'], ['favorite']), (['what', 'do', 'you', 'need', 'to', 'start'], ['how', 'do', 'i', 'start']), (['make', 'money', 'through', 'youtube'], ['earn', 'from', 'youtube']), (['when', 'it', 'comes', 'to', 'pleasure'], ['in', 'pleasure']), (['do', 'you', 'have', 'any'], ['do', 'you', 'have']), (['psychic', 'abilities'], ['psychic', 'power']), (['how', 'to', 'efficiently', 'learn'], ['how', 'to', 'learn']), (['being', 'addicted', 'to', 'porn'], ['watching', 'porn']), (['what', 'do', 'you', 'feel', 'is'], ['what', 'is']), (['purpose', 'of', 'life'], ['meaning', 'of', 'life']), (['how', 'to', 'do', 'to', 'get'], ['how', 'to', 'get']), (['what', 's'], ['what', 'is']), (['when', 'will', 'best', 'time'], ['what', 'is', 'best', 'time']), (['to', 'have', 'sex'], ['for', 'having', 'sex']), (['can', 'we', 'create'], ['how', 'to', 'make']), (["you've"], ['you', 'have']), (['important', 'to', 'world'], ['important', 'for', 'world']), (['how', 'do', 'i', 'can'], ['how', 'to']), (['in', 'your', 'opinion', 'what', 'is'], ['what', 'is']), (['song'], ['music', 'piece']), (['similar', 'to'], ['like']), (['are', 'there', 'any'], ['is', 'there', 'any']), (['is', 'there', 'any', 'other'], ['is', 'there', 'any']), (['conditions', 'inside'], ['conditions', 'in']), (['what', 'are', 'conditions'], ['how', 'are', 'conditions']), (['how', 'to', 'develop'], ['how', 'to', 'increase']), (['what', 'are', 'good', 'websites', 'to', 'learn'], ['how', 'to', 'learn']), (['english', 'speaking'], ['spoken', 'english']), (['make', 'money', 'through', 'youtube'], ['make', 'money', 'with', 'youtube']), (['is', 'there', 'any', 'way', 'to'], ['how', 'to']), (['gynecomastia'], ['man', 'boobs']), (['how', 'can', 'someone'], ['how', 'to']), (['control', 'their', 'anger'], ['control', 'anger']), (['control', 'on', 'my', 'anger'], ['control', 'anger']), (['how', 'to', 'get', 'off'], ['how', 'to', 'get', 'out', 'of']), (['how', 'do', 'get'], ['how', 'to', 'get']), (['best', 'available'], ['best']), (['technology', 'gadgets'], ['gadgets']), (['closeted', 'gay'], ['homosexual']), (['is', 'continuing'], ['continues']), (['great'], ['perfect']), (['was'], ['is']), (['mind-blowing', 'facts'], ['interesting', 'facts']), (['what', 'are', 'some', 'most'], ['what', 'are', 'some']), (['what', 'is', 'your', 'best'], ['what', 'is', 'best']), (['pitbulls'], ['pitbull']), (['questions'], ['question']), (['question', 'asked'], ['question']), (['what', 'will', 'be'], ['what', 'is']), (['what', 'can', 'i', 'do'], ['what', 'to', 'do']), (['friends'], ['friend']), (['betrayed'], ['betray'])]
    rules = []
    anchors = build_anchors(rules)

    for line in df.itertuples():
        if not line.is_duplicate:
            continue
        q1 = fix(line.question1)
        LEN1 = len(" ".join(q1)) + 5
        q2 = fix(line.question2)
        LEN2 = len(" ".join(q2)) + 5
        fmtstring = "%" + str(LEN1) + "s      %" + str(LEN2) + "s"
        print
        print
        print (">" + fmtstring) % (" ".join(q1), " ".join(q2))
        while True:
            q1set = process(stopwords, anchors, [q1])
            q2set = process(stopwords, anchors, [q2])
            q1best = None
            q2best = None
            cross = []

            match = False
            for q1str in q1set:
                q1str = " ".join(q1str)
                for q2str in q2set:
                    q2str = " ".join(q2str)
                    if q1str.find(q2str) != -1 or q2str.find(q1str) != -1 or set(q1str.split()) == set(q2str.split()):
                        match = True
                        q1best = q1str
                        q2best = q2str
                        break
                    cross.append((q1str, q2str))
                if match:
                    break

            if match:
                print ("!" + fmtstring + "      MATCH!") % (q1best, q2best)
                break
            else:
                cross = sorted([(distance(s1, s2), s1, s2) for s1, s2 in cross], reverse=True)
                for mindist, s1, s2 in cross[-7:]:
                    #mb = matching_blocks(editops(s1, s2), s1, s2)
                    #print ''.join([s1[x[0]:x[0]+x[2]] for x in mb])
                    #print ''.join([s2[x[1]:x[1]+x[2]] for x in mb])
                    print ("*" + fmtstring + "      %d") % (s1, s2, mindist)

            cmd = raw_input()
            if cmd == "":
                break
            elif cmd[0] == "+":
                stopwords.add(cmd[1:])
            elif cmd[0] == "-":
                stopwords.discard(cmd[1:])
            elif cmd == "m":
                source = raw_input().split()
                target = raw_input().split()
                if not source or not target:
                    continue
                rule = (source, target)
                rules.append(rule)
                if source[0] not in anchors:
                    anchors[source[0]] = list()
                anchors[source[0]].append(rule)
            elif cmd == "p":
                print
                print stopwords
                print
                print rules
                print anchors
                print

if __name__ == "__main__":
    main()