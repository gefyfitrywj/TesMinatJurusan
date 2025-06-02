import { defineStore } from 'pinia'

export const useResultStore = defineStore('resultStore', {
    state: () => ({
        result: null
    }),
    actions: {
        setResult(data) {
            this.result = data
        }
    }
})
