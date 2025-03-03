
from nltk.tokenize import sent_tokenize
import numpy as np
import nltk
import heapq
from langchain_openai import OpenAIEmbeddings
# nltk.download('punkt')

import os
from dotenv import load_dotenv
load_dotenv()

class TextChunker:
    def __init__(self):
        self.model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))

    def _add_context(self, sentences, window_size):
        contextualized = []
        for i in range(0, len(sentences), window_size):
            context = ' '.join(sentences[i:i+window_size])
            contextualized.append(context)
        return contextualized
    
    def process_text(self, text, windows_size=3, max_chunk_size=2000):
        sentences = sent_tokenize(text)
        chunks = self._add_context(sentences, windows_size)
        embeddings = self.model.embed_documents(chunks)
        max_chunk_count = (int(len(text) / max_chunk_size) + 1) * 2
        final_chunks = self._merge_chunks(chunks, embeddings, max_chunk_size, max_chunk_count)
        return final_chunks

    def _calculate_similarity(self, embedding1, embedding2):
        return np.dot(np.asarray(embedding1), np.asarray(embedding2))

    def update_heap_by_index(self, heap, i, new_value):
        for idx, (value, index) in enumerate(heap):
            if index == i:
                heap[idx] = heap[-1]
                heap.pop()
                heapq.heapify(heap)
                break

        heapq.heappush(heap, (new_value, i))

    def _merge_chunks(self, chunks, embeddings, max_chunk_size, max_chunk_count):
        length = len(chunks)
        breakpoints = list(range(length+1))
        similarity_pq = []
        uniques = []
        uniques_ids = []

        for i in range(0, length):
            sim_arr = []
            for j in range(0, length):
                if i == j:
                    continue
                sim_arr.append(self._calculate_similarity(embeddings[i], embeddings[j]))
            uniques.append((np.mean(sim_arr), i))
        
        uniques.sort()
        idx = 0
        while idx < 4 and idx < len(uniques):
            _, id = uniques[idx]
            uniques_ids.append(id)
            idx += 1

        for i in range(1, length):
            if i in uniques_ids or i-1 in uniques_ids:
                continue
            similarity = self._calculate_similarity(embeddings[i-1], embeddings[i])
            heapq.heappush(similarity_pq, (-similarity, i))
        
        cnt = length
        while len(similarity_pq) > 0 and cnt > max_chunk_count:
            similarity, middle = heapq.heappop(similarity_pq)
            middle_idx = breakpoints.index(middle)
            left = breakpoints[middle_idx-1]
            right = breakpoints[middle_idx+1]

            if sum(len(s) for s in chunks[left:right]) > max_chunk_size:
                continue

            if left > 0:
                pre_left = breakpoints[middle_idx-2]
                similarities = []
                for i in range(pre_left, left):
                    for j in range(left, right):
                        similarities.append(self._calculate_similarity(embeddings[i], embeddings[j]))
                
                mean_similarity = np.mean(similarities) if len(similarities) > 0 else 0

                self.update_heap_by_index(similarity_pq, left, -mean_similarity)

            if right < length:
                pre_right = breakpoints[middle_idx+2]
                similarities = []
                for i in range(right, pre_right):
                    for j in range(left, right):
                        similarities.append(self._calculate_similarity(embeddings[i], embeddings[j]))
                
                mean_similarity = np.mean(similarities) if len(similarities) > 0 else 0

                self.update_heap_by_index(similarity_pq, right, -mean_similarity)
            
            breakpoints.remove(middle)
            cnt -= 1

        ans = []
        for i in range(1, len(breakpoints)):
            ans.append(' '.join(chunks[breakpoints[i-1]:breakpoints[i]]))

        return ans
