import { useAnswerStore } from '~/stores/answerStore'

export default defineNuxtRouteMiddleware((to) => {
  const store = useAnswerStore()
  const index = parseInt(to.params.path) - 1
console.log(index)
  // Check if all previous answers are filled in
  if (!store.allPreviousAnswered(index)) {
    const redirectIndex = store.firstUnansweredIndex()
    console.log(redirectIndex)
    return navigateTo(`/quiz/${redirectIndex + 1}`)
  }
})
