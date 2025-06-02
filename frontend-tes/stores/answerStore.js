// stores/answerStore.js
import { defineStore } from 'pinia'

export const useAnswerStore = defineStore('answerStore', {
  state: () => ({
    answers: []
  }),

  actions: {
    initAnswers(length) {
      const stored = localStorage.getItem('answers')
      if (stored) {
        this.answers = JSON.parse(stored)
        // Jika panjang tidak cocok, reset ulang
        if (this.answers.length !== length) {
          this.answers = Array(length).fill(null)
        }
      } else {
        this.answers = Array(length).fill(null)
      }
      this.saveToLocal()
    },

    setAnswer(index, value) {
      this.answers[index] = value
      this.saveToLocal()
    },

    getAnswer(index) {
      return this.answers[index]
    },

    allPreviousAnswered(index) {
      return this.answers.slice(0, index).every(a => a !== null)
    },

    firstUnansweredIndex() {
      return this.answers.findIndex(a => a === null)
    },

    saveToLocal() {
      localStorage.setItem('answers', JSON.stringify(this.answers))
    },

    resetAnswers(length) {
      this.answers = Array(length).fill(null)
      this.saveToLocal()
    }
  }
})
