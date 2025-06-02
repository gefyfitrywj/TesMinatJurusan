<!-- pages/result.vue -->
<template>
    <div class="max-w-[652px] font-montserrat mx-auto pt-[157px] pb-[88px]">
        <div class="flex flex-col gap-[82px] justify-center items-center">
            <h1 class="font-bold text-[30px] leading-[40px] text-center">Hasil Tes Minat Kamu</h1>

            <div v-if="resultStore.result"
                class="bg-[#FEFEFE] w-full flex flex-col justify-center items-center gap-[24px] py-[40px] shadow-[0_4px_4px_0_#00000040] border-2 border-[#00000040] font-medium text-[30px]">
                <p>{{ resultStore.result.sma }}</p>
                <p>{{ resultStore.result.s1 }}</p>
            </div>

            <NuxtLink to="/" class="bg-[#0099FF] py-[19px] px-[30px] rounded-[8px] text-white text-center w-[70%]">
                Selesai
            </NuxtLink>
        </div>
    </div>
</template>

<script setup>
import { useResultStore } from '~/stores/resultStore'

const resultStore = useResultStore()
</script>